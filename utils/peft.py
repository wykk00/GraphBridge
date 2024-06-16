from peft import LoraConfig, PromptTuningInit, PromptTuningConfig, get_peft_model


def create_lora_config(model, rank=8):

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules = ["query", "value"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config