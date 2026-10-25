"""Render the article's two comparison tables for Medium. Requires Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import html

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagrams'
FONT='C:/Windows/Fonts/arial.ttf'
if not Path(FONT).exists(): FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def table(name,title,headers,rows,widths,row_height,foot):
    width=1800; top=170; head=100; height=top+head+row_height*len(rows)+120
    image=Image.new('RGB',(width,height),'#101b2b')
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(FONT,32); titlefont=ImageFont.truetype(FONT,48)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         f'<rect width="{width}" height="{height}" fill="#101b2b"/>']
    def write(x,y,s,f=font,color='#e5edf5'):
        draw.text((x,y),s,font=f,fill=color)
        svg.append(f'<text x="{x}" y="{y+f.size}" font-family="Arial,sans-serif" font-size="{f.size}" fill="{color}">{html.escape(s)}</text>')
    def lines(s,w):
        result=[]; line=''
        for word in s.split():
            test=(line+' '+word).strip()
            if draw.textlength(test,font=font)>w and line:
                result.append(line); line=word
            else:line=test
        if line:result.append(line)
        return result
    write(70,50,title,titlefont,'white')
    for n,row in enumerate([headers]+rows):
        y=top if n==0 else top+head+(n-1)*row_height
        rh=head if n==0 else row_height
        color='#244761' if n==0 else ('#1b2c42' if n%2 else '#152438')
        draw.rectangle((60,y,1740,y+rh),fill=color)
        svg.append(f'<rect x="60" y="{y}" width="1680" height="{rh}" fill="{color}"/>')
        x=60
        for cell,w in zip(row,widths):
            wrapped=lines(cell,w-40)
            assert len(wrapped)*42<=rh-20,(name,cell)
            for j,line in enumerate(wrapped):write(x+20,y+20+j*42,line,color='#50d6bd' if n==0 else '#e5edf5')
            x+=w
    write(70,height-75,foot,ImageFont.truetype(FONT,27),'#bbcede')
    image.save(OUT/(name+'.png'))
    svg.append('</svg>'); (OUT/(name+'.svg')).write_text('\n'.join(svg),encoding='utf8')


table('05-memory-table','70 billion parameters: a state-memory estimate',
      ['Stored state','Assumption','Approximate memory'],
      [['Weights','2 bytes per parameter','140 GB'],
       ['Gradients','2 bytes per parameter','140 GB'],
       ['Adam moment buffers','Two FP32 buffers','560 GB'],
       ['Master weights, if used','4 bytes per parameter','280 GB'],
       ['Total with master weights','16 bytes per parameter','1.12 TB']],
      [580,650,450],110,
      'Decimal units. Excludes activations, temporary buffers, fragmentation, and communication buffers.')
table('06-decision-table','Choose parallelism from the limiting resource',
      ['Observed problem','Approach to evaluate','Condition to check'],
      [['Training is slow; model fits','Data-parallel workers','Input throughput and communication cost'],
       ['Training state exceeds device memory','FSDP or another sharding strategy','Peak gathers, activations, and checkpoints'],
       ["Requests exceed one replica's capacity",'More independent replicas','A complete model fits each replica'],
       ['One inference model exceeds device or machine memory','Tensor or pipeline parallelism','Engine support, placement, and interconnect'],
       ['Independent experiments take too long','Concurrent Tune trials','Resources for every concurrent trial']],
      [570,550,560],190,
      'Choose a candidate approach, then validate memory, correctness, and performance on your workload.')
