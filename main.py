import argparse
import re
import torch
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)
from datasets import Dataset

# the config
MODEL_NAME  = "gpt2"
SAVED_MODEL = "./codegen_model"
MAX_LENGTH  = 128
EPOCHS      = 25
BATCH_SIZE  = 2
GRAD_ACCUM  = 2
DATA_REPEAT = 30
EOT         = "<|endoftext|>"

def wrap(code: str) -> str:
    return f"{EOT}\n{code.strip()}\n{EOT}"

#training data
_RAW = [
    "function add(a, b)\n\treturn a + b\nend",
    "function multiply(x, y)\n\treturn x * y\nend",
    "function greet(name)\n\tprint('Hello, ' .. name)\nend",
    "function isEven(n)\n\treturn n % 2 == 0\nend",
    "function clamp(value, lo, hi)\n\treturn math.max(lo, math.min(hi, value))\nend",
    "if x > 0 then\n\tprint('positive')\nelseif x == 0 then\n\tprint('zero')\nelse\n\tprint('negative')\nend",
    "for i = 1, 10 do\n\tprint(i)\nend",
    "while health > 0 do\n\thealth = health - 1\nend",
    "function factorial(n)\n\tif n <= 1 then\n\t\treturn 1\n\tend\n\treturn n * factorial(n - 1)\nend",
    "function fibonacci(n)\n\tif n <= 1 then return n end\n\treturn fibonacci(n - 1) + fibonacci(n - 2)\nend",
    "function listSum(t)\n\tlocal total = 0\n\tfor _, v in ipairs(t) do\n\t\ttotal = total + v\n\tend\n\treturn total\nend",
    "function tableContains(t, value)\n\tfor _, v in ipairs(t) do\n\t\tif v == value then return true end\n\tend\n\treturn false\nend",
    "function deepCopy(original)\n\tlocal copy = {}\n\tfor k, v in pairs(original) do\n\t\tif type(v) == 'table' then\n\t\t\tcopy[k] = deepCopy(v)\n\t\telse\n\t\t\tcopy[k] = v\n\t\tend\n\tend\n\treturn copy\nend",
    "local Animal = {}\nAnimal.__index = Animal\n\nfunction Animal.new(name, sound)\n\tlocal self = setmetatable({}, Animal)\n\tself.name = name\n\tself.sound = sound\n\treturn self\nend\n\nfunction Animal:speak()\n\tprint(self.name .. ' says ' .. self.sound)\nend",
    "local Counter = {}\nCounter.__index = Counter\n\nfunction Counter.new(start)\n\treturn setmetatable({ value = start or 0 }, Counter)\nend\n\nfunction Counter:increment()\n\tself.value = self.value + 1\nend\n\nfunction Counter:get()\n\treturn self.value\nend",
    "local Vehicle = {}\nVehicle.__index = Vehicle\n\nfunction Vehicle.new(name, speed)\n\tlocal self = setmetatable({}, Vehicle)\n\tself.name = name\n\tself.speed = speed\n\treturn self\nend\n\nfunction Vehicle:describe()\n\tprint(self.name .. ' goes ' .. self.speed .. ' mph')\nend",
    "local Players = game:GetService('Players')\n\nPlayers.PlayerAdded:Connect(function(player)\n\tprint(player.Name .. ' joined')\nend)\n\nPlayers.PlayerRemoving:Connect(function(player)\n\tprint(player.Name .. ' left')\nend)",
    "local ReplicatedStorage = game:GetService('ReplicatedStorage')\nlocal remoteEvent = ReplicatedStorage:WaitForChild('MyEvent')\n\nremoteEvent.OnServerEvent:Connect(function(player, data)\n\tprint(player.Name .. ' sent: ' .. tostring(data))\nend)",
    "local Players = game:GetService('Players')\nlocal player = Players.LocalPlayer\nlocal character = player.Character or player.CharacterAdded:Wait()\nlocal humanoid = character:WaitForChild('Humanoid')\n\nhumanoid.Died:Connect(function()\n\tprint(player.Name .. ' died')\nend)",
    "task.spawn(function()\n\tfor i = 1, 5 do\n\t\tprint('tick', i)\n\t\ttask.wait(1)\n\tend\nend)",
    "local function countdown(n)\n\tfor i = n, 1, -1 do\n\t\tprint(i)\n\t\ttask.wait(1)\n\tend\n\tprint('Go!')\nend\ntask.spawn(countdown, 5)",
    "local ok, err = pcall(function()\n\treturn tonumber('abc') + 1\nend)\n\nif not ok then\n\twarn('Error: ' .. err)\nend",
    "local part = Instance.new('Part')\npart.Size = Vector3.new(4, 1, 4)\npart.Position = Vector3.new(0, 5, 0)\npart.Anchored = true\npart.Parent = workspace",
    "local part = workspace.KillBrick\n\npart.Touched:Connect(function(hit)\n\tlocal humanoid = hit.Parent:FindFirstChildWhichIsA('Humanoid')\n\tif humanoid then\n\t\thumanoid.Health = 0\n\tend\nend)",
    "local Players = game:GetService('Players')\n\nPlayers.PlayerAdded:Connect(function(player)\n\tlocal leaderstats = Instance.new('Folder')\n\tleaderstats.Name = 'leaderstats'\n\tleaderstats.Parent = player\n\n\tlocal coins = Instance.new('IntValue')\n\tcoins.Name = 'Coins'\n\tcoins.Value = 0\n\tcoins.Parent = leaderstats\nend)",
    "local prompt = Instance.new('ProximityPrompt')\nprompt.ActionText = 'Collect'\nprompt.Parent = workspace.Chest\n\nprompt.Triggered:Connect(function(player)\n\tplayer.leaderstats.Coins.Value += 10\nend)",
    "local DataStoreService = game:GetService('DataStoreService')\nlocal playerData = DataStoreService:GetDataStore('PlayerData')\n\nlocal function saveData(player, coins)\n\tlocal key = 'player_' .. player.UserId\n\tpcall(playerData.SetAsync, playerData, key, coins)\nend",
    "local MathUtils = {}\n\nfunction MathUtils.lerp(a, b, t)\n\treturn a + (b - a) * t\nend\n\nfunction MathUtils.round(n)\n\treturn math.floor(n + 0.5)\nend\n\nreturn MathUtils",
    "function trim(s)\n\treturn s:match('^%s*(.-)%s*$')\nend",
    "function split(str, sep)\n\tlocal result = {}\n\tfor part in str:gmatch('[^' .. sep .. ']+') do\n\t\ttable.insert(result, part)\n\tend\n\treturn result\nend",
    "local TweenService = game:GetService('TweenService')\nlocal info = TweenInfo.new(2, Enum.EasingStyle.Sine)\nlocal tween = TweenService:Create(workspace.MyPart, info, { Position = Vector3.new(0, 20, 0) })\ntween:Play()",
]

TRAIN_TEXTS = [wrap(t) for t in _RAW] * DATA_REPEAT

# output trimmer 
_OPENS  = re.compile(r'\b(function|do|then|else|repeat)\b')
_CLOSES = re.compile(r'\bend\b')
_PYTHON = re.compile(
    r'\bnp\.\w+|\bimport\s|\bdef\s|\b__\w+__|\bpandas\b|[\x80-\xff]',
    re.MULTILINE,
)

def trim_output(text: str) -> str:
    clean = []
    for line in text.split('\n'):
        if _PYTHON.search(line):
            break
        clean.append(line)
    text = '\n'.join(clean)

    lines, out, depth, started = text.split('\n'), [], 0, False
    for line in lines:
        s      = line.strip()
        opens  = len(_OPENS.findall(s))
        closes = len(_CLOSES.findall(s))

        if s in ('end)', 'end),'):
            out.append(line)
            break

        if depth == 0 and started and s and not s.startswith('--'):
            if not re.match(r'^(function |local |if |for |while |task\.|[A-Z])', s):
                break

        depth += opens - closes
        out.append(line)
        if started and depth <= 0:
            break
        if opens > 0:
            started = True

    return re.split(r'\n{3,}', '\n'.join(out).rstrip())[0].rstrip()

    
#model
def load_model(path):
    print(f"Loading from: {path}")
    tok   = GPT2TokenizerFast.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id
    return tok, model

def load_base():
    tok   = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id
    gpu = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Loaded gpt2 base. Device: {gpu}")
    return tok, model


# dataset
def build_dataset(texts, tok):
    def tokenize(examples):
        enc = tok(examples["text"], truncation=True,
                  padding="max_length", max_length=MAX_LENGTH)
        enc["labels"] = [
            [(t if t != tok.pad_token_id else -100) for t in ids]
            for ids in enc["input_ids"]
        ]
        return enc
    return Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"])

#train
def train(tok, model, lr, schedule):
    dataset = build_dataset(TRAIN_TEXTS, tok)
    gpu     = torch.cuda.is_available()
    steps   = (len(dataset) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    print(f"Samples: {len(dataset)}  Steps: ~{steps}  LR: {lr}")
    print(f"ETA: ~{round(steps / (3.6 if gpu else 0.31) / 60)} min\n")

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            logging_steps=100,
            save_steps=9999,
            save_total_limit=1,
            learning_rate=lr,
            lr_scheduler_type=schedule,
            warmup_ratio=0.03,
            weight_decay=0.0,
            fp16=gpu,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    ).train()

    model.save_pretrained(SAVED_MODEL)
    tok.save_pretrained(SAVED_MODEL)
    print(f"\nSaved to {SAVED_MODEL}")

# generate
def generate(model, tok, prompt, max_new_tokens=100):
    model.eval()
    device     = next(model.parameters()).device
    inp        = tok(f"{EOT}\n{prompt}", return_tensors="pt").to(device)
    prompt_len = inp["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
            # eos_token_id omitted - trimmer handles stopping
        )

    completion = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    return trim_output(prompt + completion.split(EOT)[0])

    
DEMOS = {
    "factorial":            "function factorial(n)\n\tif n <= 1 then",
    "vehicle OOP":          "local Vehicle = {}\nVehicle.__index = Vehicle\n\nfunction Vehicle.new(name, speed)\n\tlocal self =",
    "touched / kill brick": "local part = workspace.KillBrick\n\npart.Touched:Connect(function(hit)\n\tlocal humanoid =",
    "leaderstats":          "local Players = game:GetService('Players')\n\nPlayers.PlayerAdded:Connect(function(player)\n\tlocal leaderstats =",
    "task countdown":       "local function countdown(n)\n\tfor i = n, 1, -1 do",
    "pcall":                "local ok, err = pcall(function()\n\treturn",
    "fibonacci":            "function fibonacci(n)\n\tif n <= 1 then return n end",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Fresh retrain from gpt2 with LR=1e-4 (recommended)")
    parser.add_argument("--train", action="store_true",
                        help="Continue from checkpoint with LR=5e-6")
    args = parser.parse_args()

    if args.retrain:
        tok, model = load_base()
        train(tok, model, lr=1e-4, schedule="cosine")
    elif args.train:
        try:
            tok, model = load_model(SAVED_MODEL)
        except Exception:
            tok, model = load_base()
        train(tok, model, lr=5e-6, schedule="constant")
    else:
        try:
            tok, model = load_model(SAVED_MODEL)
        except Exception:
            print("No saved model — retraining from scratch.\n")
            tok, model = load_base()
            train(tok, model, lr=1e-4, schedule="cosine")
            tok, model = load_model(SAVED_MODEL)

    print("\n── DEMOS ─────────────────────────────────────────────")
    for label, prompt in DEMOS.items():
        print(f"\n[{label}]")
        print(generate(model, tok, prompt))
        print("─" * 50)

if __name__ == "__main__":
    main()
