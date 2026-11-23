from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import torch
import timeimport copy, os, random, threading, time
import chromadb
from duckduckgo_search import DDGS
from google.colab import drive
import torch, torch.nn as nn
import gradio as gr

# 1. Stream 40,000 high-quality educational texts from FineWeb-Edu
print('Downloading FineWeb-Edu sample...')
dataset = load_dataset(
    'HuggingFaceFW/fineweb-edu',
    name='sample-10BT',
    split='train',
    streaming=True,
)
texts = [item['text'] for item in dataset.take(40000)]

# 2. Train an 8192-vocab BPE tokenizer with chat special tokens

print('Training custom 8k tokenizer...')
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ['<|endoftext|>', '<|user|>', '<|assistant|>', '<|pad|>']
trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save('custom_8k_tokenizer.json')
# 3. Tokenize the pre-training dataset into a single tensor
print('Tokenizing pre-training data...')
eot_id = tokenizer.token_to_id('<|endoftext|>')
all_tokens = []
for t in texts:
  all_tokens.extend(tokenizer.encode(t).ids)
  all_tokens.append(eot_id)

train_data = torch.tensor(all_tokens, dtype=torch.long)
print(f'Ready! Total pre-training tokens: {len(train_data):,}')
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):

   def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

   def forward(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class ModernAttention(nn.Module):

   def __init__(self, config):
    super().__init__()
    self.n_head = config['n_head']
    self.n_kv_head = config['n_kv_head']
    self.head_dim = config['n_embd'] // config['n_head']
    self.n_rep = self.n_head // self.n_kv_head

    self.wq = nn.Linear(
         config['n_embd'], self.n_head * self.head_dim, bias=False
    )
    self.wk = nn.Linear(
         config['n_embd'], self.n_kv_head * self.head_dim, bias=False
    )
    self.wv = nn.Linear(
        config['n_embd'], self.n_kv_head * self.head_dim, bias=False
    )
    self.wo = nn.Linear(
         self.n_head * self.head_dim, config['n_embd'], bias=False
    )
   def forward(self, x, freqs_cis):
    B, T, _ = x.size()
    xq = self.wq(x).view(B, T, self.n_head, self.head_dim)
    xk = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
    xv = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    xk = xk.repeat_interleave(self.n_rep, dim=2)
    xv = xv.repeat_interleave(self.n_rep, dim=2)

    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    y = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))

class SwiGLU(nn.Module):

   def __init__(self, config):
      super().__init__()
      hidden_dim = int(2 * (4 * config['n_embd']))
      self.w1 = nn.Linear(config['n_embd'], hidden_dim, bias=False)
      self.w2 = nn.Linear(hidden_dim, config['n_embd'], bias=False)
      self.w3 = nn.Linear(config['n_embd'], hidden_dim, bias=False)

   def forward(self, x):
      return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):

   def __init__(self, config):
    super().__init__()
    self.attn_norm = RMSNorm(config['n_embd'])
    self.attn = ModernAttention(config)
    self.ffn_norm = RMSNorm(config['n_embd'])
    self.feed_forward = SwiGLU(config)

   def forward(self, x, freqs_cis):
    x = x + self.attn(self.attn_norm(x), freqs_cis)
    return x + self.feed_forward(self.ffn_norm(x))


class MiniLlama(nn.Module):

   def __init__(self, config):
    super().__init__()
    self.config = config
    self.tok_embeddings = nn.Embedding(config['vocab_size'], config['n_embd'])
    self.layers = nn.ModuleList(
        [LlamaBlock(config) for _ in range(config['n_layer'])]
    )
    self.norm = RMSNorm(config['n_embd'])
    self.output = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
    self.tok_embeddings.weight = self.output.weight

    head_dim = config['n_embd'] // config['n_head']
    freqs_cis = precompute_freqs_cis(head_dim, config['block_size'] * 2)
    self.register_buffer('freqs_cis', freqs_cis, persistent=False)
   def forward(self, idx, targets=None):
     B, T = idx.size()
     x = self.tok_embeddings(idx)
     for layer in self.layers:
       x = layer(x, self.freqs_cis[:T])
     logits = self.output(self.norm(x))

     loss = None
     if targets is not None:
       loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
     return logits, loss

   @torch.no_grad()
   def generate(self, idx, max_new_tokens, temperature=0.7, stop_token_id=None):
    for _ in range(max_new_tokens):
       idx_cond = idx[:, -self.config['block_size'] :]
       logits, _ = self(idx_cond)
       logits = logits[:, -1, :] / temperature
       probs = F.softmax(logits, dim=-1)
       idx_next = torch.multinomial(probs, num_samples=1)
       if stop_token_id is not None and idx_next.item() == stop_token_id:
          break
       idx = torch.cat((idx, idx_next), dim=1)
    return idx


config = {
    'vocab_size': 8192,
    'block_size': 512,  # Longer 512-token context window
    'n_layer': 12,  # 12 deep layer
    'n_head': 8,  # 8 Query heads
    'n_kv_head': 2,  # 2 Key/Value
    'n_embd': 480,  # ~40.3M total parameters
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniLlama(config).to(device)
print(
    'Total parameters:'
    f' {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M'
)



def get_batch(data_tensor, batch_size=24, block_size=512):
  ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
  x = torch.stack([data_tensor[i : i + block_size] for i in ix]).to(device)
  y = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in ix]).to(
      device
  )
  return x, y


max_steps = 4000
optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=5e-5
)
scaler = torch.cuda.amp.GradScaler()

model.train()
start_time = time.time()
for step in range(max_steps):
  xb, yb = get_batch(train_data)
  optimizer.zero_grad(set_to_none=True)

  with torch.cuda.amp.autocast(dtype=torch.float16):
      _, loss = model(xb, yb)

  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  scaler.step(optimizer)
  scaler.update()
  scheduler.step()

  if step % 400 == 0:
    print(
        f'Pretrain Step {step}/{max_steps} | Loss: {loss.item():.4f} | Time:'
        f' {time.time() - start_time:.1f}s'
    )

torch.save(model.state_dict(), 'minillama_40m_base.pt')
print('Base model saved!')
# 1. Load clean instruction pairs
sft_dataset = load_dataset(
    'HuggingFaceTB/smoltalk', 'everyday-conversations', split='train'
)

sft_tokens = []
for row in sft_dataset:
  conversation = ''
  for msg in row['messages']:
    role_tag = '<|user|>' if msg['role'] == 'user' else '<|assistant|>'
    conversation += f"{role_tag}\n{msg['content']}\n"
  conversation += '<|endoftext|>'
  sft_tokens.extend(tokenizer.encode(conversation).ids)

sft_data = torch.tensor(sft_tokens, dtype=torch.long)

# 2. Fine-tune with a lower learning rate so it keeps its pre-trained knowledge
sft_steps = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
model.train()
for step in range(sft_steps):
  xb, yb = get_batch(sft_data, batch_size=16, block_size=512)
  optimizer.zero_grad(set_to_none=True)
  with torch.cuda.amp.autocast(dtype=torch.float16):
    _, loss = model(xb, yb)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()

  if step % 200 == 0:
    print(f'Chat SFT Step {step}/{sft_steps} | Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'minillama_40m_chat.pt')
print('Chat model saved!')


drive.mount('/content/drive')
BRAIN_DIR = '/content/drive/MyDrive/InfiniteBrain_40M'
os.makedirs(BRAIN_DIR, exist_ok=True)

db = chromadb.PersistentClient(
         path=f'{BRAIN_DIR}/db'
).get_or_create_collection('mem')
buf, lock, last_active = [], threading.Lock(), time.time()


def web_search(q, n=2):
    try:
       with DDGS() as d:
          return '\n'.join(
                f"{r['title']}: {r['body']}" for r in d.text(q, max_results=n)
          )
    except Exception:
      return ''


def recall_mem(q):
    return (
         '\n'.join(
                db.query(query_texts=[q], n_results=min(2, db.count()))[
                      'document'
                ]
         )
         if db.count()
         else ''
    )
tok = globals().get('tokenizer', globals().get('enc'))
dev = next(model.parameters()).device
bs = (
    model.config.get('block_size', 256)
    if isinstance(model.config, dict)
    else getattr(model.config, 'block_size', 256)
)

if os.path.exists(f'{BRAIN_DIR}/w.pt'):
  try:
      model.load_state_dict(
            torch.load(f'{BRAIN_DIR}/w.pt', map_location=dev), strict=False
      )
  except Exception:
      pass

def params_m():
  return sum(p.numel() for p in model.parameters()) / 1e6


def enc(t):
  return (
      tok.encode_ordinary(t)
      if hasattr(tok, 'encode_ordinary')
      else (tok.encode(t).ids if hasattr(tok.encode(t), 'ids') else tok.encode(t))
  )


def remember_mem(t):
   db.add(documents=[t], ids=[f'{time.time()}_{random.random()}'])
   buf.extend(enc(t + '\n'))
def grow_brain():
   if params_m() >= 125.0:
      return
   layers = getattr(
       model, 'layers', getattr(getattr(model, 'transformer', None), 'h', None)
   )
   if layers:
      blk = copy.deepcopy(layers[-1]).to(dev)
      for n, p in blk.named_parameters():
          if any(k in n for k in ['wo.weight', 'w2.weight', 'c_proj.weight']):
            nn.init.zeros_(p)
      layers.append(blk)
      torch.save(model.state_dict(), f'{BRAIN_DIR}/w.pt')

def learn_step(steps=6, lr=1e-4):
  if len(buf) < bs + 2:
    return
  t = torch.tensor(buf[-100000:], dtype=torch.long)
  opt = torch.optim.AdamW(model.parameters(), lr=lr)
  model.train()
  for _ in range(steps):
     ix = torch.randint(len(t) - bs - 1, (4,))
     xb = torch.stack([t[i : i + bs] for i in ix]).to(dev)
     yb = torch.stack([t[i + 1 : i + bs + 1] for i in ix]).to(dev)
     opt.zero_grad(set_to_none=True)
     _, loss = model(xb, yb)
     loss.backward()
     opt.step()
def enhance_prompt(prompt):
  if not prompt.strip():
    return prompt
  idx = torch.tensor(
      [enc(f'Expand visual scene: {prompt}\nScene:')], device=dev
  )
  with lock:
      model.eval()
      out = model.generate(idx, max_new_tokens=30)
  return (
      f'{prompt}, {tok.decode(out[idx.shape:].tolist()).strip()}, cinematic, 8k'
  )


def _idle():
  c = 0
  while True:
    if time.time() - last_active > 20:
      f = web_search(
          random.choice(['science', 'history', 'AI', 'space', 'coding']), 2
      )
      if f:
        remember_mem(f)
      with lock:
        learn_step(8, 8e-5)
        c += 1
        if c % 5 == 0:
          grow_brain()
          time.sleep(8)


threading.Thread(target=_idle, daemon=True).start()
print(f'Brain ready! Current size: {params_m():.1f}M parameters')
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers.utils import export_to_video

img_pipe = vid_pipe = None


def make_image(p):
  global img_pipe
  if not p.strip():
    return None
  with lock:
    if not img_pipe:
      img_pipe = AutoPipelineForText2Image.from_pretrained(
          'stabilityai/sd-turbo', torch_dtype=torch.float16, variant='fp16'
      )
      img_pipe.enable_model_cpu_offload()
    return img_pipe(p, num_inference_steps=2, guidance_scale=0.0).images


def make_video(p):
  global vid_pipe
  if not p.strip():
    return None
  with lock:
    if not vid_pipe:
      vid_pipe = DiffusionPipeline.from_pretrained(
          'ali-vilab/text-to-video-ms-1.7b',
          torch_dtype=torch.float16,
          variant='fp16',
      )
      vid_pipe.enable_model_cpu_offload()
      vid_pipe.enable_vae_slicing()
    export_to_video(
        vid_pipe(p, num_inference_steps=20, num_frames=16).frames,
        'out.mp4',
        fps=8,
    )
    return 'out.mp4'



def chat_fn(msg, hist, use_web):
  global last_active
  last_active = time.time()
  ctx = (
      f"{recall_mem(msg)}\n{web_search(msg) if use_web else ''}\nUser:"
      f' {msg}\nAssistant:'
  )
  idx = torch.tensor([enc(ctx)], device=dev)
  with lock:
    model.eval()
    reply = tok.decode(
        model.generate(idx, max_new_tokens=100)[idx.shape :].tolist()
    ).strip()
    remember_mem(f'Q: {msg}\nA: {reply}')
    learn_step(4, 1e-4)
  last_active = time.time()
  hist.append((msg, reply))
  return (
      '',
      hist,
       (
           f'**Params:** {params_m():.1f}M / 125M | **Infinite Memories:**'
           f' {db.count():,}'
       ),
  )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
  gr.Markdown('# 🚀 Self-Evolving AI Studio')
  with gr.Tabs():
    with gr.TabItem('💬 Chat'):
      stat = gr.Markdown(
          f'**Params:** {params_m():.1f}M / 125M | **Infinite Memories:**'
          f' {db.count():,}'
      )
      web_cb = gr.Checkbox(label='Live Web Search', value=True)
      bot = gr.Chatbot(height=350)
      txt = gr.Textbox(placeholder='Ask anything...')
      txt.submit(chat_fn, [txt, bot, web_cb], [txt, bot, stat])
    with gr.TabItem('🎨 Image'):
      ip = gr.Textbox(label='Prompt')
      gr.Button('✨ Enhance').click(enhance_prompt, ip, ip)
      gr.Button('Generate', variant='primary').click(
          make_image, ip, gr.Image()
      )
    with gr.TabItem('🎬 Video'):
      vp = gr.Textbox(label='Prompt')
      gr.Button('✨ Enhance').click(enhance_prompt, vp, vp)
      gr.Button('Generate', variant='primary').click(
          make_video, vp, gr.Video()
      )

demo.launch(share=True)
