import importlib

# Initialize prompts
def initialize_prompts(kwargs):
    for k, v in kwargs.items():
        if k == 'prompt_config':
            for kk, vv in v.items():
                if isinstance(vv, str) and vv.startswith('prompts.'):
                    prompt_module = importlib.import_module('.'.join(vv.split('.')[:-2]))
                    prompt_class = getattr(prompt_module, vv.split('.')[-2])()
                    prompt = getattr(prompt_class, vv.split('.')[-1])
                    kwargs[k][kk] = prompt
    return kwargs

# Save Prompts Function
def save_prompts(*prompts):
    with open("./prompts/PREFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[0] == '' or prompts[0][-1] != '\n':
            writer.write(prompts[0])
        else:
            writer.write(prompts[0]+'\n')
    with open("./prompts/SUFFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[1] == '' or prompts[1][-1] != '\n':
            writer.write(prompts[1])
        else:
            writer.write(prompts[1]+'\n')
    with open("./prompts/RETRIEVEPREFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[2] == '' or prompts[2][-1] != '\n':
            writer.write(prompts[2])
        else:
            writer.write(prompts[2]+'\n')
    with open("./prompts/RETRIEVESUFFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[3] == '' or prompts[3][-1] != '\n':
            writer.write(prompts[3])
        else:
            writer.write(prompts[3]+'\n')
    with open("./prompts/AIPREFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[4] == '' or prompts[4][-1] != '\n':
            writer.write(prompts[4])
        else:
            writer.write(prompts[4]+'\n')
    print("Prompts save completed")
