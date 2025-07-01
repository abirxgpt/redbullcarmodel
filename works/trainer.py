# Import necessary libraries for the custom trainer implementation
import os  # Operating system interface for file and directory operations
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules and functions

# Import base Trainer class and related utilities from transformers
from transformers import Trainer  # Base trainer class from Hugging Face
from transformers.trainer import (
    is_sagemaker_mp_enabled,  # Check if SageMaker model parallelism is enabled
    get_parameter_names,  # Utility to get parameter names from model
    ALL_LAYERNORM_LAYERS,  # List of all layer normalization layer types
    is_peft_available,  # Check if PEFT (Parameter Efficient Fine-Tuning) is available
    WEIGHTS_NAME,  # Standard name for model weights file
    TRAINING_ARGS_NAME,  # Standard name for training arguments file
    SAFE_WEIGHTS_NAME,  # Standard name for safetensors weights file
    TRAINER_STATE_NAME,  # Standard name for trainer state file
    PREFIX_CHECKPOINT_DIR,  # Prefix for checkpoint directory names
    logger,  # Logger for training messages
)
import safetensors  # Library for safe tensor serialization
from peft import PeftModel  # PEFT model wrapper for parameter efficient fine-tuning
from typing import Optional  # Type hinting for optional parameters
import numpy as np  # Numerical computing library
from transformers.processing_utils import ProcessorMixin  # Base class for processors
from transformers.modeling_utils import PreTrainedModel  # Base class for pre-trained models
from peft import PeftModel  # PEFT model wrapper (imported again for clarity)
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3  # Custom utilities for PEFT state management

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Handle parameter extraction in DeepSpeed ZeRO-3 environment.
    ZeRO-3 partitions model parameters across multiple GPUs, so this function
    gathers the parameter data when needed and converts it to CPU for processing.
    
    Args:
        param: The parameter tensor to process
        ignore_status: Whether to ignore the parameter's availability status
        name: Optional name of the parameter for debugging
    
    Returns:
        Parameter tensor on CPU as a detached clone
    """
    # Import DeepSpeed ZeRO utilities
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    # Check if parameter is managed by DeepSpeed ZeRO
    if hasattr(param, "ds_id"):
        # Check if parameter is currently available
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")  # Debug message for unavailable parameters
        
        # Use DeepSpeed's context manager to gather the parameter from all GPUs
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()  # Get data, detach from graph, move to CPU, and clone
    else:
        # For non-ZeRO parameters, simply detach and move to CPU
        param = param.detach().cpu().clone()
    
    return param

class Phi3VTrainer(Trainer):
    """
    Custom trainer class for Phi-3 Vision model fine-tuning.
    Extends the base Hugging Face Trainer with specific functionality for:
    - Multi-component learning rates (vision, projection, language model)
    - LoRA-specific checkpoint saving
    - Processor handling for multimodal inputs
    """

    def __init__(self, *args, processor: Optional[ProcessorMixin] = None, **kwargs):
        """
        Initialize the Phi3VTrainer with an optional processor.
        
        Args:
            *args: Variable positional arguments passed to parent Trainer
            processor: Optional processor for handling multimodal inputs (text + images)
            **kwargs: Variable keyword arguments passed to parent Trainer
        """
        # Initialize the parent Trainer class
        super(Phi3VTrainer, self).__init__(*args, **kwargs)
        # Store the processor for later use in saving and processing
        self.processor = processor

    def create_optimizer(self):
        """
        Create a custom optimizer with different learning rates for different model components.
        This allows fine-grained control over training dynamics:
        - Vision model can have a different (usually lower) learning rate
        - Image projection layer can have its own learning rate
        - Language model components use the default learning rate
        
        Returns:
            Configured optimizer with parameter groups
        """
        # Check if SageMaker model parallelism is enabled - if so, use default behavior
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        # Get the model to optimize (could be wrapped by PEFT)
        opt_model = self.model

        # Only create optimizer if it doesn't already exist
        if self.optimizer is None:
            # Get parameter names that should have weight decay applied
            # Typically excludes bias terms and layer normalization parameters
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Create mapping of module keywords to their specific learning rates
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["img_projection"] = self.args.projector_lr  # Custom LR for image projection layer
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr  # Custom LR for vision model
            
            # If we have custom learning rates, create separate parameter groups
            if len(lr_mapper) > 0:
                # Find parameters that should use special learning rates
                special_lr_parameters = [
                    name for name, _ in opt_model.named_parameters() 
                    if any(module_keyword in name for module_keyword in lr_mapper)
                ]
                
                # Create parameter groups for default learning rate
                optimizer_grouped_parameters = [
                    {
                        # Parameters with weight decay (excluding special LR parameters)
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        # Parameters without weight decay (excluding special LR parameters)
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                
                # Add parameter groups for each special learning rate
                for module_keyword, lr in lr_mapper.items():
                    # Find all parameters belonging to this module
                    module_parameters = [
                        name for name, _ in opt_model.named_parameters() 
                        if module_keyword in name
                    ]
                    
                    # Add parameter groups with custom learning rate
                    optimizer_grouped_parameters.extend([
                        {
                            # Parameters with weight decay and custom LR
                            "params": [
                                p for n, p in opt_model.named_parameters() 
                                if (n in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,  # Custom learning rate
                        },
                        {
                            # Parameters without weight decay and custom LR
                            "params": [
                                p for n, p in opt_model.named_parameters() 
                                if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": lr,  # Custom learning rate
                        },
                    ])
            else:
                # No custom learning rates - use standard parameter grouping
                optimizer_grouped_parameters = [
                    {
                        # Parameters with weight decay
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        # Parameters without weight decay
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            # Get optimizer class and kwargs from training arguments
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Create the optimizer with the grouped parameters
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # Handle special case for 8-bit Adam optimizer
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                # Get the global optimization manager for 8-bit training
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                # Skip embedding layers from 8-bit optimization (keep them in FP32)
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        # Count parameters in embedding layers
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        # Register override to use 32-bit optimization for embeddings
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save model checkpoint with special handling for LoRA models.
        For LoRA models, we need to save both the LoRA weights and the non-LoRA weights separately.
        
        Args:
            model: The model to save
            trial: Hyperparameter tuning trial information
            metrics: Training metrics for determining best model
        """
        # Special handling for LoRA-enabled models
        if self.args.lora_enable:
            # Create checkpoint directory name with global step
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            # Store floating point operations count if not doing hyperparameter search
            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            # Determine output directory
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Save the model (LoRA weights)
            self.save_model(output_dir, _internal_call=True)

            # Extract and save non-LoRA weights (base model parameters that may have been updated)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), 
                require_grad_only=False
            )
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            # Save additional training state if not saving only the model
            if not self.args.save_only_model:
                # Save optimizer state (Adam momentum, etc.)
                self._save_optimizer_and_scheduler(output_dir)
                # Save random number generator state for reproducibility
                self._save_rng_state(output_dir)

            # Update best model tracking based on metrics
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                # Ensure metric name has 'eval_' prefix
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                # Determine if higher or lower values are better
                operator = np.greater if self.args.greater_is_better else np.less
                
                # Update best model if this is the first checkpoint or if metric improved
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save trainer state (step count, best metrics, etc.)
            if self.args.should_save:
                # Update control state for callbacks
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                # Save state to JSON file
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            # Push to Hugging Face Hub if configured
            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Clean up old checkpoints if configured
            if self.args.should_save:
                # Use checkpoint number (not modification time) for rotation
                # This is more reliable in cloud environments
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            # For non-LoRA models, use the standard checkpoint saving
            super(Phi3VTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save the model, tokenizer, processor, and training arguments.
        This method handles the final model saving with special considerations for
        multimodal models and different model types.
        
        Args:
            output_dir: Directory to save the model to
            state_dict: Optional custom state dictionary to save
        """
        # Use provided output directory or default from training arguments
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Define supported model classes for saving
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        
        # Handle model saving based on model type
        if not isinstance(self.model, supported_classes):
            # For non-standard model types, save state dictionary
            if state_dict is None:
                state_dict = self.model.state_dict()

            # Check if the unwrapped model (removing DDP/FSDP wrappers) is supported
            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                # Save using the model's save_pretrained method
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, 
                    state_dict=state_dict, 
                    safe_serialization=self.args.save_safetensors
                )
            else:
                # Fallback: save only the state dictionary
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    # Save using safetensors format (recommended for security)
                    safetensors.torch.save_file(
                        state_dict, 
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME), 
                        metadata={"format": "pt"}
                    )
                else:
                    # Save using standard PyTorch format
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # For supported model types, use save_pretrained
            if state_dict is None:
                state_dict = self.model.state_dict()
            
            # Filter out 'wte' (word token embeddings) parameters if present
            # This might be specific to certain model architectures
            state_dict = {k: v for k, v in state_dict.items() if "wte" not in k}
            
            # Save the model using its save_pretrained method
            self.model.save_pretrained(
                output_dir, 
                state_dict=state_dict, 
                safe_serialization=self.args.save_safetensors
            )

        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save processor if available (important for multimodal models)
        if self.processor is not None:
            # Sync chat template from tokenizer to processor
            self.processor.chat_template = self.processor.tokenizer.chat_template
            # Save the processor configuration
            self.processor.save_pretrained(output_dir)

        # Save training arguments for reproducibility and model loading
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # Commented out debugging function for training step
    # This function can be uncommented to debug which parameters are being trained
    # def training_step(self, model, inputs):
    #     """
    #     Debug function to print which parameters are being trained.
    #     Useful for verifying that the correct model components are being updated.
    #     """
    #     # Check vision model parameters
    #     for name, param in model.named_parameters():
    #         if 'vision_model' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
            
    #         # Check image projection parameters
    #         elif 'img_projection' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #     
    #     # Call parent training step
    #     return super().training_step(model, inputs)
