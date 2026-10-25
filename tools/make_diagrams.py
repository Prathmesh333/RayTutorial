"""Render editable, deterministic diagrams. Requires Pillow, not a Ray dependency."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import html

OUT = Path(__file__).resolve().parents[1] / "diagrams"
OUT.mkdir(exist_ok=True)
FONT = "C:/Windows/Fonts/arial.ttf"
if not Path(FONT).exists():
    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def create(name, title, subtitle, nodes, edges, notes):
    im = Image.new("RGB", (1800, 1000), "#101b2b")
    d = ImageDraw.Draw(im)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1000" viewBox="0 0 1800 1000">',
           '<rect width="1800" height="1000" fill="#101b2b"/>']
    def text(x,y,s,size=28,color="#c7d7e8",anchor="la"):
        d.text((x,y),s,font=ImageFont.truetype(FONT,size),fill=color,anchor=anchor)
        svg.append(f'<text x="{x}" y="{y+size if anchor=="la" else y+size/3}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{"middle" if anchor=="mm" else "start"}">{html.escape(s)}</text>')
    text(80,60,title,52,"white")
    text(80,145,subtitle,27)
    for x,y,w,h,label,sub in nodes:
        d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill="#1b3550",outline="#6585a7",width=3)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#1b3550" stroke="#6585a7" stroke-width="3"/>')
        text(x+w/2,y+h/2-18,label,30,"white","mm")
        text(x+w/2,y+h/2+30,sub,23,"#c7d7e8","mm")
    for a,b,label in edges:
        d.line([a,b],fill="#50d6bd",width=4)
        t=math.atan2(b[1]-a[1],b[0]-a[0])
        pts=[b,(b[0]-18*math.cos(t-.4),b[1]-18*math.sin(t-.4)),(b[0]-18*math.cos(t+.4),b[1]-18*math.sin(t+.4))]
        d.polygon(pts,fill="#50d6bd")
        svg.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#50d6bd" stroke-width="4"/>')
        svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in pts)+'" fill="#50d6bd"/>')
        if label:text((a[0]+b[0])/2,(a[1]+b[1])/2-28,label,24,"#50d6bd","mm")
    for i,n in enumerate(notes):text(80,850+50*i,n,27)
    im.save(OUT/(name+'.png'))
    svg.append('</svg>'); (OUT/(name+'.svg')).write_text('\n'.join(svg),encoding='utf8')


create('01-cluster','One cluster, several machines','Conceptual control and data paths; resource counts are illustrative.',
 [(80,260,400,170,'Driver / job','Submits tasks and actors'),(700,260,410,170,'Head node','GCS + cluster services'),
  (80,560,600,170,'Worker node A','Raylet + object store + CPU workers'),(1000,560,650,170,'Worker node B','Raylet + object store + GPU workers')],
 [((480,345),(700,345),'Submit'),((905,430),(380,560),'Coordinate'),((905,430),(1325,560),'Coordinate'),((680,645),(1000,645),'Object transfer')],
 ['Raylets schedule work locally and coordinate placement across nodes.', 'PyTorch or an inference engine performs tensor communication; Ray schedules the workers.'])
create('02-workflow','One small model, two inference paths','The companion learns y = 3x + 2 from 1,024 synthetic rows.',
 [(80,280,350,150,'Ray Data','Create + preprocess'),(660,280,470,150,'Ray Train + PyTorch','Shards -> synchronized training'),(1350,280,370,150,'Checkpoint store','Model + optimizer + epoch'),
  (280,620,480,150,'Ray Data actors','Offline batch predictions'),(1080,620,480,150,'Ray Serve replica','HTTP predictions')],
 [((430,355),(660,355),'Batches'),((1130,355),(1350,355),'Save'),((1430,430),(520,620),'Load weights'),((1560,430),(1320,620),'Load weights')],
 ['Single-machine demo: local checkpoint directory. Multi-node jobs: shared durable storage.', 'Tune runs separate model trials; selecting a trial is separate from distributed training.'])
create('03-training','Data parallelism and model sharding','Adding workers does not automatically reduce the model state required on each GPU.',
 [(80,270,720,110,'Replicated data parallelism','Each GPU holds a full model copy'),(1000,270,720,110,'Fully sharded data parallelism','Persistent model state is partitioned'),
  (100,490,300,180,'GPU 0','Full model; batch A'),(480,490,300,180,'GPU 1','Full model; batch B'),
  (1020,490,300,180,'GPU 0','State shard A'),(1400,490,300,180,'GPU 1','State shard B')],
 [((400,580),(480,580),''),((1320,580),(1400,580),'')],
 ['DDP synchronizes gradients. FSDP gathers parameters as needed and synchronizes shards.', 'Tensor parallelism partitions operations. Pipeline parallelism partitions sequences of layers.'])
create('04-serving','Replicas and a distributed model solve different problems','A logical model replica can itself require several GPUs and machines.',
 [(80,260,700,110,'Independent replicas','Each replica has a complete model'),(980,260,740,110,'One partitioned replica','All partitions cooperate on inference'),
  (90,490,320,170,'Replica A','Complete model'),(480,490,320,170,'Replica B','Complete model'),
  (1000,490,320,170,'Node A: 4 GPUs','Pipeline stage 0; TP=4'),(1400,490,320,170,'Node B: 4 GPUs','Pipeline stage 1; TP=4')],
 [((1320,575),(1400,575),'')],
 ['Left: route each request to one replica. Right: stage 0 sends activations to stage 1.', 'Right: 4 tensor-parallel workers x 2 pipeline stages = 8 GPUs per replica; placement must match.'])
