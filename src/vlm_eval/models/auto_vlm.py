from src.config.settings import VLM_MODEL_CONFIG, VLM_TYPE_TO_CLASS

def get_vlm_client(model_name):
    cfg = VLM_MODEL_CONFIG[model_name]
    model_type = cfg.get("type")

    if model_type not in VLM_TYPE_TO_CLASS:
        raise ValueError(f"Unknown model type: {model_type}")

    client_class = VLM_TYPE_TO_CLASS[model_type]
    
    # Prepare constructor arguments from config, excluding 'type'
    client_args = {k: v for k, v in cfg.items() if k != 'type'}
    
    return client_class(**client_args) 