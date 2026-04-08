"""
prompt_templates.py
===================
DisasterM3 task-specific prompt templates.
Adapted from the original authors' prompts in DisasterM3/pyscripts/run_vllm.py.

Returns structured data so each VLM handler can insert images in its native format.
"""

# Tasks that inherently require pre + post disaster images
DUAL_IMAGE_TASKS = {
    "Disaster Bearing Bodies Recognition",
    "Building Damage Counting",
    "Disaster Type Recognition",
    "Road Damage Counting",
    "Disaster Report",
    "Disaster Restoration Advice",
}


def get_prompt_and_images(sample: dict) -> dict:
    """
    Build the task-specific prompt and determine which images to pass.

    Parameters
    ----------
    sample : dict
        A sample from dataset_loader.load_disasterm3_bench().

    Returns
    -------
    dict with prompt_text, image_paths, needs_dual_image.
    """
    task = sample["task_type"]
    prompt_raw = sample["prompt_raw"]
    options_str = sample["options_str"]
    needs_dual = task in DUAL_IMAGE_TASKS

    # ── Resolve image paths ──────────────────────────────────────
    if needs_dual:
        pre = sample.get("pre_image_path")
        post = sample.get("post_image_path")
        if pre and post:
            image_paths = [pre, post]
        else:
            image_paths = [sample["primary_image_path"]]
            needs_dual = False
    elif task == "Disaster Scene Recognition":
        # Landuse — uses ONLY the pre-disaster image
        pre = sample.get("pre_image_path")
        image_paths = [pre] if pre else [sample["primary_image_path"]]
    else:
        image_paths = [sample["primary_image_path"]]

    # ── Build prompt text (NO image placeholders) ────────────────
    if task == "Disaster Bearing Bodies Recognition":
        prompt_text = (
            "Analyze both the pre-disaster and post-disaster images to answer "
            "the following question. Choose the best option(s) from the candidate "
            "options provided.\n\n"
            f"Question: {prompt_raw}\n"
            f"Options: {options_str}\n\n"
            "Your task is to respond with ONLY the capital letters of the correct "
            "options, separated by a comma and a space (e.g., C, D, H). "
            "Do not include any explanation or other text."
        )

    elif task in ("Building Damage Counting", "Disaster Type Recognition",
                  "Road Damage Counting"):
        prompt_text = (
            "Analyze both the pre-disaster and post-disaster images to answer "
            "the following question. Choose the best option from the candidate "
            "options provided.\n\n"
            f"Question: {prompt_raw}\n"
            f"Options: {options_str}\n\n"
            "Your task is to respond with ONLY the capital letter of the correct "
            "option (e.g., C). Do not include any explanation or other text."
        )

    elif task == "Disaster Scene Recognition":
        prompt_text = (
            "Analyze the image to answer the following question. Choose the best "
            "option(s) from the candidate options provided.\n\n"
            f"Question: {prompt_raw}\n"
            f"Options: {options_str}\n\n"
            "Your task is to respond with ONLY the capital letters of the correct "
            "options, separated by a comma and a space (e.g., C, D, H). "
            "Do not include any explanation or other text."
        )

    elif task == "Relational Reasoning":
        prompt_text = (
            "Analyze the image to answer the following question. Choose the best "
            "option from the candidate options provided.\n\n"
            f"Question: {prompt_raw}\n"
            f"Options: {options_str}\n\n"
            "Your task is to respond with ONLY the capital letter of the correct "
            "option (e.g., C). Do not include any explanation or other text."
        )

    elif task == "Disaster Report":
        prompt_text = (
            "Your TASK is to analyze the provided pair of pre-disaster and "
            "post-disaster remote sensing images. You will act as a remote sensing "
            "analyst to identify the type of disaster and assess its impact on both "
            "built and natural environments across five specific categories.\n\n"
            "Your analysis must be formatted as follows:\n"
            "DISASTER: [the name of the disaster]\n"
            "BUILDING: [describe impacts on buildings]\n"
            "ROAD: [describe impacts on road networks]\n"
            "VEGETATION: [describe impacts on natural, unmanaged vegetation cover]\n"
            "WATER_BODY: [describe changes to water bodies]\n"
            "AGRICULTURE: [describe impacts on managed agricultural land]\n"
            "CONCLUSION: [provide a concise 1-2 sentence summary synthesizing the "
            "overall disaster impacts observed across the categories.]"
        )

    elif task == "Disaster Restoration Advice":
        prompt_text = (
            "Your TASK is to generate concise and integrated recovery recommendations "
            "for the affected area based on the provided pre-disaster and post-disaster "
            "remote sensing images. Aspects to focus on include infrastructure "
            "restoration, housing reconstruction, and ecological and geological "
            "environment restoration.\n\n"
            "Based on your analysis of the images:\n"
            "1. First determine if recovery actions are necessary. If no significant "
            "damage or impact is observed, clearly state no recovery recommendations "
            "due to no discernible impact.\n"
            "2. If recovery is needed, provide recommendations in the following format:\n"
            "IMMEDIATE_RECOVERY: [Provide an integrated paragraph within 50 words "
            "describing immediate recovery actions. Create a flowing narrative.]\n"
            "LONG_TERM_RECOVERY: [Provide an integrated paragraph within 50 words "
            "describing long-term recovery strategies. Create a flowing narrative.]\n\n"
            "Ensure your recommendations are realistic, feasible, and properly "
            "prioritized based on the visible damage in the images."
        )

    else:
        # Generic fallback
        prompt_text = prompt_raw
        if options_str:
            prompt_text += f"\nOptions: {options_str}\nAnswer with the correct option letter(s)."

    return {
        "prompt_text": prompt_text,
        "image_paths": image_paths,
        "needs_dual_image": needs_dual,
    }
