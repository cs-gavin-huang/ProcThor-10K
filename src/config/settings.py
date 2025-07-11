# src/config/settings.py

from typing import List, Dict


from src.vlm_eval.models.vlm_client import OllamaClient, OpenAIClient

# VLM模型配置
VLM_MODELS = {
    "llava-vllm": "liuhaotian/llava-v1.5-7b",
    "gemma3-1b": "gemma3:1b",
    "gemma3-4b": "gemma3:4b"
    # "qwen2.5vl-3b": "qwen2.5vl:3b",
    # "mistral-small": "mistral-small3.1:24b",
    # "llava-7b": "llava:7b",
    # "llava-13b": "llava:13b",
    # "llama3-vision": "llama3.2-vision:11b",
    # "minicpm-v": "minicpm-v:8b",
    # "llava-llama3": "llava-llama3:8b",
    # "moondream": "moondream:1.8b",
    # "granite-vision": "granite3.2-vision:2b",
    # "gemma3-12b": "gemma3:12b",
    # "qwen2.5vl-7b": "qwen2.5vl:7b"
}


VLM_MODEL_CONFIG = {
    "gemma3-1b": {
        "type": "ollama",
        "model_id": "gemma3:1b"
    },
    "gemma3-4b": {
        "type": "ollama",
        "model_id": "gemma3:4b"
    },
    "llava-vllm": {
        "type": "vllm",
        "model_id": "liuhaotian/llava-v1.5-7b",
        "api_base": "http://localhost:30000"
    }
    # "qwen2.5vl-3b": {
    #     "type": "ollama",
    #     "model_id": "qwen2.5vl:3b"
    # },
    # "mistral-small": {
    #     "type": "ollama",
    #     "model_id": "mistral-small3.1:24b"
    # },
    # "llava-7b": {
    #     "type": "ollama",
    #     "model_id": "llava:7b"
    # },
    # "llava-13b": {
    #     "type": "ollama",
    #     "model_id": "llava:13b"
    # },
    # "llama3-vision": {
    #     "type": "ollama",
    #     "model_id": "llama3.2-vision:11b"
    # },
    # "minicpm-v": {
    #     "type": "ollama",
    #     "model_id": "minicpm-v:8b"
    # },
    # "llava-llama3": {
    #     "type": "ollama",
    #     "model_id": "llava-llama3:8b"
    # },
    # "moondream": {
    #     "type": "ollama",
    #     "model_id": "moondream:1.8b"
    # },
    # "granite-vision": {
    #     "type": "ollama",
    #     "model_id": "granite3.2-vision:2b"
    # },
    # "gemma3-12b": {
    #     "type": "ollama",
    #     "model_id": "gemma3:12b"
    # },
    # "qwen2.5vl-7b": {
    #     "type": "ollama",
    #     "model_id": "qwen2.5vl:7b"
    # }
}


EXPERIMENT_CONFIG = {
    "sample_sizes": [10, 100, "full"],
    "agent_types": ["standard", "LoCoBot"],
    
    "prompt_groups": {
        "affordance": {
            "properties": ["moveable", "pickupable"],
            "prompts": {
                "v1": "Please determine if the {object_type} object at a distance of {distance} meters can be {property}. Please choose between true or false and output only one word.",
                "v2": "You are a {agent_type} robot with limited pushing capabilities. Please determine if the {object_type} object at a distance of {distance} meters can be {property}. Please choose between true or false and output only one word.",
                "pv01": "Your response must be 'true' or 'false' only. You are a {agent_type} robot. As a {agent_type} robot, you have specific physical characteristics and capabilities. For example:\\n\\n*   If you are a **standard** agent, you can typically lift objects around 1-2 kilograms, like a book or a small appliance, and push objects up to 10-15 kilograms, like a chair or a small table.\\n*   If you are a **LoCoBot**, you are a relatively small mobile robot. You can typically lift objects around 0.5 to 1 kilogram, such as a can of soda or a small tool, and push objects around 5 to 10 kilograms, like a lightweight stool or a small box.\\n\\nGiven these specific capabilities, and considering the current object is {object_type} at a distance of {distance} meters.\\nPlease determine if you can perform the {property} operation on this {object_type} object.\\nPlease only answer 'true' or 'false'. Output only 'true' or 'false'.",
                "pv02": "Your response must be 'true' or 'false' only. Task Description: Based on robot type, object characteristics, distance, and specified action, determine if the robot can perform the action.\\n\\nExample 1:\\nRobot Type: Small\\nObject: A standard apple\\nDistance: 0.5 meters\\nAction: pickupable\\nJudgment: true\\n\\nExample 2:\\nRobot Type: Small\\nObject: A household refrigerator\\nDistance: 1 meter\\nAction: moveable\\nJudgment: false\\n\\nExample 3:\\nRobot Type: Large\\nObject: An office chair\\nDistance: 0.2 meters\\nAction: pickupable\\nJudgment: true\\n\\nExample 4:\\nRobot Type: Medium\\nObject: A 10-liter water bucket filled with water\\nDistance: 0.3 meters\\nAction: moveable\\nJudgment: true\\n\\nYour task:\\nRobot Type: {agent_type}\\nObject: {object_type}\\nDistance: {distance} meters\\nAction: {property}\\nJudgment: Output only 'true' or 'false'.",
                "pv03": "Your response must be 'true' or 'false' only. You are a {agent_type} robot.\\nThere is a {object_type} object {distance} meters away from you.\\nYou need to determine if you can perform the {property} operation on this object.\\n\\nPlease think step by step about the following questions:\\n1. As a {agent_type} robot, what physical capabilities and limitations do I typically have related to performing the '{property}' action? (For example, for 'pickupable', consider my load capacity, arm reach range, and gripper design; for 'moveable', consider my pushing force, pulling force, or movement method.)\\n2. What physical characteristics does this {object_type} object typically have? (For example, estimate its weight, size, shape, material, fragility, etc.)\\n3. Based on my capabilities, the object's characteristics, and the current distance of {distance} meters, can I successfully perform the '{property}' action?\\n\\nBased on the above reasoning, please determine if you can perform the {property} operation on this {object_type} object.\\nPlease only answer 'true' or 'false'. Output only 'true' or 'false'.",
                "pv04": "Your response must be 'true' or 'false' only. Task Description: Based on robot type, object characteristics, distance, and specified action, determine if the robot can perform the action through step-by-step reasoning.\\n\\nExample 1:\\nRobot Type: Small\\nObject: A glass cup\\nDistance: 0.3 meters\\nAction: pickupable\\nReasoning Process: Small robots typically have dexterous grippers suitable for grasping small objects. A glass cup is light and small. 0.3 meters is within the typical arm reach range of a small robot. Therefore, it can be picked up.\\nJudgment: true\\n\\nExample 2:\\nRobot Type: Small\\nObject: A heavy desk\\nDistance: 0.1 meters\\nAction: moveable\\nReasoning Process: Small robots typically have limited pushing force. Although the desk is close, its weight far exceeds the pushing capability of a small robot. Therefore, it cannot be moved.\\nJudgment: false\\n\\nExample 3:\\nRobot Type: Large\\nObject: A medium-sized cardboard box (about 10kg)\\nDistance: 1 meter\\nAction: pickupable\\nReasoning Process: Large robots typically have strong load capacity. A 10kg cardboard box is within their load range. 1 meter is also reachable for large robots. Assuming their gripper is suitable for grasping cardboard boxes. Therefore, it can be picked up.\\nJudgment: true\\n\\nYour task:\\nRobot Type: {agent_type}\\nObject: {object_type}\\nDistance: {distance} meters\\nAction: {property}\\nReasoning Process: [Model generates reasoning process here]\\nJudgment: Output only 'true' or 'false'.",
                "pv05": "Your response must be 'true' or 'false' only. About Robot Affordance Judgment:\\n\\nPart 1: Knowledge Generation\\nPlease first briefly describe what core physical capabilities, related sensor configurations, and potential key limitations a {agent_type} robot typically has when attempting to perform actions like '{property}'.\\n\\nPart 2: Affordance Judgment\\nNow, based on your above description of the {agent_type} robot's capabilities and limitations for the '{property}' action, please determine:\\nIf there is a {object_type} object located {distance} meters away from this robot, can the robot perform the {property} operation on this {object_type} object?\\n\\nPlease only answer 'true' or 'false'. Output only 'true' or 'false'."
            }
        },
        "identification": {
            "properties": ["identification"],
            "prompts": {
                "v1": "What is the object type of the object at a distance of {distance} meters? Please output only the object's category name as a single word.",
                "v2": "You are a {agent_type} robot. Please identify the category of the object at a distance of {distance} meters. Output only a single word representing the category.",
                "pv01": "Your response must be a single-word object category. You are a {agent_type} robot with advanced visual sensors. Your task is to perform object recognition. Given the object presented to you, what is its primary category? Output only the category name.",
                "pv02": "Your response must be a single-word object category.\nExample 1:\nImage: [An image of an apple]\nJudgment: Apple\n\nExample 2:\nImage: [An image of a sofa]\nJudgment: Sofa\n\nYour task:\nImage: [Current image]\nJudgment:",
                "pv03": "Your response must be a single-word object category.\nPlease think step by step about the following questions:\n1. What are the key visual features of the object (e.g., shape, color, texture, typical size)?\n2. What context is the object in (e.g., kitchen, office)?\n3. Based on these features and context, what is the most likely category for this object?\n\nBased on the above reasoning, please state the object's category.\nOutput only the category name.",
                "pv04": "Your response must be a single-word object category.\nExample 1:\nImage: [An image of a laptop on a desk]\nReasoning Process: The object has a screen and a keyboard connected by a hinge. It is a portable computer. This is characteristic of a laptop.\nJudgment: Laptop\n\nExample 2:\nImage: [An image of a single red rose]\nReasoning Process: The object has a green stem, thorns, and red petals arranged in a spiral. This is a type of flower known as a rose.\nJudgment: Rose\n\nYour task:\nImage: [Current image]\nReasoning Process: [Model generates reasoning process here]\nJudgment:",
                "pv05": "Your response must be a single-word object category.\nPart 1: Visual Feature Analysis\nPlease first briefly describe the core visual characteristics of the object shown in the image, such as its shape, material, and key components.\n\nPart 2: Category Judgment\nNow, based on your analysis of the visual features, what is the most probable category of this object?\nPlease only output the category name as a single word."
            }
        },
        "state": {
            "properties": ["state_isOpen"],
            "prompts": {
                "v1": "Please determine if the {object_type} object at a distance of {distance} meters is open. Please choose between true or false and output only one word.",
                "v2": "You are a {agent_type} robot with visual sensors. Please determine if the {object_type} object at a distance of {distance} meters is open. Please choose between true or false and output only one word.",
                "pv01": "Your response must be 'true' or 'false' only. You are a {agent_type} robot. Your task is to determine the state of objects. Given that the current object is {object_type}, please determine if this {object_type} object is open. Please only answer 'true' or 'false'. Output only 'true' or 'false'.",
                "pv02": "Your response must be 'true' or 'false' only.\nExample 1:\nObject: A closed microwave\nJudgment: false\n\nExample 2:\nObject: An open laptop\nJudgment: true\n\nYour task:\nObject: {object_type}\nJudgment: Output only 'true' or 'false'.",
                "pv03": "Your response must be 'true' or 'false' only.\nThere is a {object_type} object {distance} meters away from you.\nYou need to determine if this object is currently open.\n\nPlease think step by step about the following questions:\n1. What are the visual cues for an object of type '{object_type}' being open? (e.g., a visible gap, an angled door, visible interior contents).\n2. Does the image show any of these visual cues?\n3. Based on the visual evidence, is the object open or closed?\n\nBased on the above reasoning, please determine if the object is open.\nPlease only answer 'true' or 'false'. Output only 'true' or 'false'.",
                "pv04": "Your response must be 'true' or 'false' only.\nExample 1:\nObject: A refrigerator with its door slightly ajar.\nReasoning Process: A visible gap can be seen between the door and the main body of the refrigerator. The interior light might be on. This indicates an open state.\nJudgment: true\n\nExample 2:\nObject: A closed book on a table.\nReasoning Process: The front and back covers are flush with each other. The pages are not visible. This indicates a closed state.\nJudgment: false\n\nYour task:\nObject: {object_type}\nReasoning Process: [Model generates reasoning process here]\nJudgment: Output only 'true' or 'false'.",
                "pv05": "Your response must be 'true' or 'false' only.\nAbout Object State Judgment:\n\nPart 1: Knowledge Generation\nPlease first briefly describe the typical visual indicators that distinguish an 'open' state from a 'closed' state for an object like a {object_type}.\n\nPart 2: State Judgment\nNow, based on your above description of the visual indicators, please determine:\nIs the {object_type} object shown in the image currently open?\n\nPlease only answer 'true' or 'false'. Output only 'true' or 'false'."
            }
        },
        "reachability": {
            "properties": ["reachable_moveable", "reachable_pickupable"],
            "prompts": {
                "v1": "Considering my current position, can I successfully perform the {property} action on the {object_type} at a distance of {distance} meters? Please answer with only 'true' or 'false'.",
                "v2": "You are a {agent_type} robot. From your current location, determine if the {object_type} object, which is {distance} meters away, is reachable for you to {property}. Please choose between true or false and output only one word.",
                "pv01": "Your response must be 'true' or 'false' only. You are a {agent_type} robot with a specific arm reach. The {object_type} is {distance} meters away. Given your physical limitations, can you currently reach and {property} it? Please only answer 'true' or 'false'.",
                "pv02": "Your response must be 'true' or 'false' only. Task: Determine if an object is reachable to act upon.\n\nExample 1:\nObject: A book on a faraway shelf\nDistance: 3.0 meters\nAction: pickupable\nJudgment: false\n\nYour task:\nObject: {object_type}\nDistance: {distance} meters\nAction: {property}\nJudgment: Output only 'true' or 'false'.",
                "pv03": "Your response must be 'true' or 'false' only.\nThere is a {object_type} object {distance} meters away from you.\nThink step by step: 1. Is the distance of {distance} meters within my robot's typical interaction range? 2. Are there any immediate obstacles shown in the image that would prevent me from reaching it?\nBased on this, determine if you can currently {property} the object. Answer only 'true' or 'false'.",
                "pv04": "Your response must be 'true' or 'false' only.\nExample 1:\nObject: A cup on the same table as the robot.\nDistance: 0.5 meters\nAction: pickupable\nReasoning: The distance is very short, well within the robot's arm reach. There are no obstructions. Therefore, the robot can reach it.\nJudgment: true\n\nYour task:\nObject: {object_type}\nDistance: {distance} meters\nAction: {property}\nReasoning: [Model generates reasoning process here]\nJudgment: Output only 'true' or 'false'.",
                "pv05": "Your response must be 'true' or 'false' only.\nPart 1: Assess Reachability. Based on a typical {agent_type} robot's reach and the distance of {distance} meters, is the {object_type} object within the interaction zone?\n\nPart 2: Reachability Judgment. Can you currently perform the {property} action on the object? Please only answer 'true' or 'false'."
            }
        }
    }
}

# Dynamically generate the list of all properties to be evaluated
all_properties = []
for group_data in EXPERIMENT_CONFIG["prompt_groups"].values():
    all_properties.extend(group_data["properties"])
EXPERIMENT_CONFIG["affordance_properties"] = all_properties


OUTPUT_CONFIG = {
    "base_dir": "experiment_VLM_eval_object_affordance",
    "metrics": {
        "accuracy": lambda pred, true: sum(p == t for p, t in zip(pred, true)) / len(pred),
        "f1_score": lambda pred, true: 2 * sum(p and t for p, t in zip(pred, true)) / (sum(pred) + sum(true))
    }
}


TARGET_INTERACTABLE_OBJECT_TYPES = {
    "Mug", "Television", "Chair", "Door", "Microwave", "Laptop", "Cabinet", "Fridge", "Book"
} 


GROUND_TRUTH_AFFORDANCES = {
    "Default": {"moveable": False, "pickupable": False, "state_isOpen": False}, 

    "Book": {"moveable": True, "pickupable": True},
    "Cabinet": {"moveable": False, "pickupable": False, "state_isOpen": False},
    "Chair": {"moveable": True, "pickupable": True},
    "Door": {"moveable": False, "pickupable": False, "state_isOpen": True},
    "Fridge": {"moveable": True, "pickupable": False, "state_isOpen": False},
    "Laptop": {"moveable": True, "pickupable": True, "state_isOpen": True},
    "Microwave": {"moveable": True, "pickupable": False, "state_isOpen": False},
    "Mug": {"moveable": True, "pickupable": True},
    "Television": {"moveable": True, "pickupable": False},
    "Wall": {"moveable": False, "pickupable": False}, 
    "Room": {"moveable": False, "pickupable": False}

}


VLM_TYPE_TO_CLASS = {
    "ollama": OllamaClient,
    "openai": OpenAIClient, 
}