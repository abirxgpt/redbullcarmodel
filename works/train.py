# Import necessary libraries for model training
import os  # Operating system interface for file operations
import torch  # PyTorch deep learning framework
import transformers  # Hugging Face transformers library for NLP models
from peft import LoraConfig, get_peft_model  # Parameter Efficient Fine-Tuning library for LoRA
import ast  # Abstract Syntax Trees for safely evaluating string literals
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM  # Hugging Face model components
from training.trainer import Phi3VTrainer  # Custom trainer class for Phi-3 Vision model
from training.data import make_supervised_data_module  # Data loading utilities
from training.params import DataArguments, ModelArguments, TrainingArguments  # Configuration classes
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer  # Training utility functions
import pathlib  # Object-oriented filesystem paths

# Global variable to track the local rank in distributed training
local_rank = None

def rank0_print(*args):
    """
    Print function that only outputs on rank 0 process in distributed training.
    This prevents duplicate print statements across multiple processes.
    
    Args:
        *args: Variable arguments to print
    """
    # Only print if this is the main process (rank 0) or single process training
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    """
    Identify linear and embedding layers in the model that should be targeted for LoRA adaptation.
    LoRA works by adding low-rank matrices to existing linear layers to enable efficient fine-tuning.
    
    Args:
        model: The neural network model to analyze
        num_lora_modules: Maximum number of modules to include (-1 for all)
        lora_namespan_exclude: List of module name patterns to exclude from LoRA
        verbose: Whether to print the found module names
    
    Returns:
        List of module names suitable for LoRA adaptation
    """
    # Define the types of layers that can be adapted with LoRA
    linear_cls = torch.nn.modules.Linear  # Fully connected layers
    embedding_cls = torch.nn.modules.Embedding  # Embedding layers
    lora_module_names = []  # List to store eligible module names

    # Handle special case for embedding tokens exclusion
    if 'embed_tokens' in lora_namespan_exclude:
        lora_namespan_exclude.remove('embed_tokens')
        lora_namespan_exclude += ['model.embed_tokens']  # Add full path

    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        # Skip modules that match exclusion patterns
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        # Check if module is a linear or embedding layer
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    # Limit the number of modules if specified (take the last N modules)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    
    # Print found modules if verbose mode is enabled
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    """
    Set the requires_grad attribute for a collection of parameters.
    This controls whether gradients are computed for these parameters during backpropagation.
    
    Args:
        parameters: Iterable of PyTorch parameters
        requires_grad: Boolean indicating whether gradients should be computed
    """
    # Loop through all parameters and set their gradient computation flag
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    """
    Configure the vision tower (image encoder) component of the multimodal model.
    This sets up the vision model's data type, device placement, and gradient requirements.
    
    Args:
        model: The multimodal model containing the vision tower
        training_args: Training configuration arguments
        compute_dtype: Data type for computations (float16, bfloat16, or float32)
        device: Device to place the model on (CPU or GPU)
    """
    # Get the vision model from the vision embedding component
    vision_tower = model.vision_embed_tokens.img_processor.vision_model
    # Move vision tower to specified device and data type
    vision_tower.to(dtype=compute_dtype, device=device)

    # Configure the image projection layer (maps vision features to text embedding space)
    img_projection_params = model.vision_embed_tokens.img_projection.parameters()
    # Enable/disable gradients based on whether we're tuning the projector
    set_requires_grad(img_projection_params, training_args.tune_img_projector)

    # Configure the vision model parameters
    vision_model_params = vision_tower.parameters()
    # Enable gradients only if vision tower is not frozen
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Handle quantized training (4-bit or 8-bit)
    if training_args.bits in [4, 8]:
        # Move image processor to specified device and data type for quantized training
        model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=device)

def configure_llm(model, training_args):
    """
    Configure the language model (LLM) component of the multimodal model.
    This sets up gradient requirements for different parts of the language model.
    
    Args:
        model: The language model to configure
        training_args: Training configuration arguments
    """
    # Configure the language model head (final layer that produces token probabilities)
    lm_head_params = model.lm_head.parameters()
    # Enable gradients only if LLM is not frozen
    set_requires_grad(lm_head_params, not training_args.freeze_llm)

    # Configure the token embedding layer
    embed_token_params = model.model.embed_tokens.parameters()
    # Enable gradients only if LLM is not frozen
    set_requires_grad(embed_token_params, not training_args.freeze_llm)

    # Configure transformer layers and normalization layers
    for name, param in model.model.named_parameters():
        # Check if parameter belongs to transformer layers or normalization
        if name.startswith('layers') or name.startswith('norm'):
            # Enable gradients only if LLM is not frozen
            param.requires_grad = not training_args.freeze_llm

def train():
    """
    Main training function that orchestrates the entire fine-tuning process.
    This function handles model loading, configuration, data preparation, and training execution.
    """
    # Access the global local_rank variable
    global local_rank

    # Parse command line arguments using Hugging Face argument parser
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    # Parse arguments into dataclass instances
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Validate that num_crops doesn't exceed the maximum supported value
    assert training_args.num_crops <= 16, 'num_crops must be less than or equal to 16'

    # Validate LoRA configuration: if LoRA is enabled, LLM must be frozen
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    # Validate vision LoRA configuration: if vision LoRA is enabled, vision tower must be frozen
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    # Validate that vision LoRA is only used when general LoRA is enabled
    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
    else:
        # Parse LoRA exclusion list from string if provided
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        # If vision LoRA is disabled, exclude vision components from LoRA adaptation
        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["vision_model", "img_projection"]

    # Set local rank for distributed training
    local_rank = training_args.local_rank
    
    # Determine compute data type based on training arguments
    compute_dtype = (torch.float16 if training_args.fp16 else 
                    (torch.bfloat16 if training_args.bf16 else torch.float32))

    # Configure quantization settings for 4-bit or 8-bit training
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},  # Map all model components to specified device
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,  # Enable 4-bit quantization
                load_in_8bit=training_args.bits==8,  # Enable 8-bit quantization
                llm_int8_skip_modules=["img_projection", "vision_model"],  # Skip quantizing vision components
                llm_int8_threshold=6.0,  # Threshold for outlier detection in 8-bit quantization
                llm_int8_has_fp16_weight=False,  # Use int8 weights instead of fp16
                bnb_4bit_compute_dtype=compute_dtype,  # Data type for 4-bit computations
                bnb_4bit_use_double_quant=training_args.double_quant,  # Enable double quantization
                bnb_4bit_quant_type=training_args.quant_type,  # Quantization algorithm type
            )
        ))

    # Load the pre-trained model with specified configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,  # Model identifier from Hugging Face Hub
        torch_dtype=compute_dtype,  # Set model data type
        cache_dir=training_args.cache_dir,  # Directory for caching downloaded models
        trust_remote_code=True,  # Allow execution of custom code in the model
        _attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",  # Attention implementation
        **bnb_model_from_pretrained_args  # Quantization configuration
    )

    # Disable caching during training to save memory
    model.config.use_cache = False

    # Configure model for quantized training
    if training_args.bits in [4,8]:
        # Set torch dtype in config for quantized models
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else 
                                   (torch.bfloat16 if training_args.bf16 else torch.float32))
        # Prepare model for k-bit training with gradient checkpointing
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True}  # Workaround for gradient checkpointing bug
        )
    
    # Enable gradient checkpointing if specified (trades compute for memory)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # Enable gradients for input embeddings
        # Set gradient checkpointing kwargs with reentrant mode for compatibility
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Configure LoRA if enabled
    if training_args.lora_enable:
        # Get the list of modules to exclude from LoRA
        lora_namespan_exclude = training_args.lora_namespan_exclude
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=training_args.lora_rank,  # Rank of the low-rank matrices
            lora_alpha=training_args.lora_alpha,  # Scaling parameter for LoRA
            target_modules=find_target_linear_names(
                model, 
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules
            ),  # Modules to apply LoRA to
            lora_dropout=training_args.lora_dropout,  # Dropout rate for LoRA layers
            bias=training_args.lora_bias,  # How to handle bias parameters
            task_type="CAUSAL_LM",  # Task type for causal language modeling
        )
        
        # Convert model to appropriate precision for 16-bit training
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        # Add LoRA adapters to the model
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    # Load the processor for handling text and image inputs
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,  # Same model ID as the main model
        cache_dir=training_args.cache_dir,  # Cache directory
        padding_side='right',  # Pad sequences on the right side
        trust_remote_code=True,  # Allow custom code execution
        num_crops=training_args.num_crops,  # Number of image crops for data augmentation
        model_max_length=training_args.max_seq_length  # Maximum sequence length
    )
    
    # Configure tokenizer padding settings
    processor.tokenizer.pad_token = processor.tokenizer.unk_token  # Use unknown token for padding
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)  # Get padding token ID
    processor.tokenizer.padding_side = 'right'  # Pad on the right side

    # Store tokenizer configuration in model config
    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    
    # Configure the model components based on LoRA settings
    if training_args.lora_enable:
        # When using LoRA, access the underlying model
        model_to_configure = model.model
        configure_llm(model_to_configure, training_args)
    else:
        # For full fine-tuning, configure the model directly
        model_to_configure = model.model
        configure_llm(model_to_configure, training_args)
    
    # Configure vision tower only if vision LoRA is not being used
    if not training_args.vision_lora:
        configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
        
    # Store learning rates in model config for different components
    model.config.vision_lr = training_args.vision_lr  # Learning rate for vision components
    model.config.projector_lr = training_args.projector_lr  # Learning rate for projection layers

    # Handle mixed precision training for quantized models
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        # Iterate through all modules to set appropriate data types
        for name, module in model.named_modules():
            # Convert LoRA layers to bfloat16 if specified
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            
            # Keep normalization layers in float32 for stability
            if 'norm' in name:
                module = module.to(torch.float32)
            
            # Handle language model head and embedding layers
            if ('lm_head' in name or 'embed_token' in name) and 'vision_embed_token' not in name:
                if hasattr(module, 'weight'):
                    # Convert to bfloat16 if using bf16 and weight is float32
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Create data module for supervised fine-tuning
    data_module = make_supervised_data_module(
        processor=processor,  # Processor for handling inputs
        data_args=data_args   # Data configuration arguments
    )

    # Initialize the custom trainer for Phi-3 Vision
    trainer = Phi3VTrainer(
        model=model,          # The model to train
        processor=processor,  # Input processor
        args=training_args,   # Training configuration
        **data_module        # Training and validation datasets
    )

    # Check if there are existing checkpoints and resume training if found
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)  # Resume from latest checkpoint
    else:
        trainer.train()  # Start training from scratch

    # Save the final training state
    trainer.save_state()

    # Re-enable caching after training
    model.config.use_cache = True
    
    # Save the model based on training configuration
    if training_args.lora_enable:
        # For LoRA training, save LoRA weights and non-LoRA parameters separately
        
        # Extract LoRA adapter weights
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), 
            training_args.lora_bias
        )

        # Extract non-LoRA parameters (base model weights that may have been updated)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), 
            require_grad_only=False
        )

        # Save model configuration and weights on main process only
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)  # Save model configuration
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)  # Save LoRA weights
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))  # Save non-LoRA weights
    else:
        # For full fine-tuning, save the entire model
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

# Entry point for script execution
if __name__ == "__main__":
    train()  # Start the training process
