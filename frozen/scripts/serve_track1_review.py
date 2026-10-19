"""Read-only local review for the thirty Track 1 test sequences and produced videos."""
import argparse
import html
import json
import mimetypes
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse

import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track1_safe_windows_v1';DATA=OUT/'review';UI=ROOT/'debug_viewer/track1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8767);a=p.parse_args()
    inventory=json.loads((OUT/'inventory.json').read_text());episodes={ep['episode'] for ep in inventory};arrays=OrderedDict()
    def overview():
        status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {'episodes':{}}
        notes=json.loads((OUT/'review_notes.json').read_text()) if (OUT/'review_notes.json').exists() else {}
        stages={'object':'物体网格重建','depth':'单目深度','human_export':'人体输入导出','human':'人体估计',
            'alignment':'人体尺度对齐','scale':'物体尺度校准','metric_export':'重建输入导出','foundationpose':'物体位姿跟踪',
            'coconet':'CoCoNet 推理','refinement':'CARI4D 300 步优化','canonical':'MHR 参数验证',
            'optimizer_input':'固定人体参数','tracker':'候选轨迹跟踪','safe_windows':'safe_windows_v1 窗口优化',
            'review_export':'可视化导出','submission_validation':'提交数据校验'}
        rows=[];done=0
        for ep in inventory:
            name=ep['episode'];r=ROOT/'outputs'/name/'safe_windows_v1/report.json';s=status['episodes'].get(name,{})
            if r.exists():
                report=json.loads(r.read_text());ready=report.get('status')=='complete'
            else:ready=False
            if ready:
                done+=1;state='完成';score=f"{report['baseline']['total']:.3f} → {report['after']['total']:.3f}"
                links=f'<a href="/viewer#episode={name}&frame=0&view=full">3D 对照</a> · <a href="/media/{name}/comparison.mp4">对比视频</a> · <a href="/media/{name}/comparison.png">关键帧</a>'
            else:
                state=html.escape({'segmented':'分割完成','running':'运行中','failed':'待处理 / 补跑'}.get(s.get('status'),s.get('status','等待')))
                score=html.escape(stages.get(s.get('stage'),s.get('stage','')));links=''
            mask=OUT/'segmentation_review'/name/'masks_preview.mp4'
            if mask.exists():links+=f' · <a href="/media/{name}/masks_preview.mp4">分割视频</a>'
            note=html.escape(notes.get(name,{}).get('note',''))
            if notes.get(name,{}).get('status')=='needs_visual_review':
                note='<strong style="color:#a14d12">重点检查：</strong>'+note
            rows.append(f'<tr><td>{ep["episode_index"]:02d}</td><td>{html.escape(ep["object"])}</td><td>{ep["frames"]}</td><td>{state}</td><td>{score}</td><td>{links}</td><td>{note}</td></tr>')
        return ('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Track 1 · 30 条测试审核</title><style>body{font:16px system-ui;background:#f5f5f0;color:#26352f;max-width:1280px;margin:40px auto;padding:0 24px}h1{font-size:30px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#087969}p{line-height:1.7}.note{background:#e6ede5;padding:16px}</style>'''
            +f'<h1>Track 1 · safe_windows_v1</h1><p>已完成 {done}/30 条 · 16,563 个输入帧 · 9,877 个正式评分帧</p>'
            +'<p class="note">先检查三栏视频，再用 3D 对照旋转视角检查深度和接触。人体与尺度固定；当前表格为训练目标，不是正式成绩。所有提交文件等待审核，尚未上传。</p>'
            +'<p><a href="/submission_validation.json">提交文件校验</a> · <a href="/status.json">运行状态</a> · <a href="/implementation.json">固定配置</a></p>'
            +'<table><thead><tr><th>序列</th><th>物体</th><th>帧数</th><th>状态</th><th>目标 / 当前阶段</th><th>查看</th><th>审核备注</th></tr></thead><tbody>'
            +''.join(rows)+'</tbody></table><p>页面每 60 秒刷新，独立视频及 3D 查看页不自动刷新。</p><script>setTimeout(()=>location.reload(),60000)</script></html>').encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            url=urlparse(self.path);parts=url.path.strip('/').split('/');kind='application/octet-stream'
            try:
                path=None
                if url.path in ['/','/overview']:data=overview();kind='text/html; charset=utf-8'
                elif url.path=='/api/frame':
                    q=parse_qs(url.query);ep=q['episode'][0];stage=q['stage'][0];frame=int(q['frame'][0])
                    if ep not in episodes or stage not in ['baseline','ours']:raise ValueError()
                    key=(ep,stage)
                    if key not in arrays:arrays[key]=np.load(DATA/ep/(stage+'.npy'),mmap_mode='r')
                    if len(arrays)>8:arrays.popitem(last=False)
                    values=arrays[key]
                    if not 0<=frame<len(values):raise ValueError()
                    data=values[frame].tobytes()
                elif url.path in ['/viewer','/index.html','/app.js','/style.css']:
                    path=UI/('index.html' if url.path in ['/viewer','/index.html'] else parts[0])
                elif url.path=='/catalog.json':path=DATA/'catalog.json'
                elif url.path in ['/status.json','/submission_validation.json','/implementation.json','/review_notes.json','/media_validation.json','/mesh_quality.json']:path=OUT/parts[0]
                elif len(parts)==3 and parts[0]=='data' and parts[1] in episodes and parts[2] in ['manifest.json','object.json','body_faces.bin','hand_faces.bin']:
                    path=DATA/parts[1]/parts[2]
                elif len(parts)==3 and parts[0]=='rgb' and parts[1] in episodes and len(parts[2])==10 and parts[2][:6].isdigit() and parts[2].endswith('.jpg'):
                    path=ROOT/'outputs'/parts[1]/'segmentation/frames'/parts[2]
                elif len(parts)==3 and parts[0]=='media' and parts[1] in episodes and parts[2] in ['comparison.mp4','comparison.png','masks_preview.mp4']:
                    path=(OUT/'segmentation_review'/parts[1]/parts[2] if parts[2]=='masks_preview.mp4'
                        else ROOT/'outputs'/parts[1]/'safe_windows_v1'/parts[2])
                elif parts[0]=='vendor':
                    base=(ROOT/'debug_viewer/node_modules/three').resolve();path=(base/'/'.join(parts[1:])).resolve()
                    if not path.is_relative_to(base) or path.suffix!='.js':raise FileNotFoundError()
                else:raise FileNotFoundError()
                if path is not None:
                    if not path.is_file():raise FileNotFoundError()
                    kind=mimetypes.guess_type(path.name)[0] or kind
                    if path.suffix=='.mp4':
                        size=path.stat().st_size;start,end=0,size-1;range_header=self.headers.get('Range')
                        if range_header:
                            unit,span=range_header.split('=',1)
                            if unit!='bytes' or ',' in span:raise ValueError()
                            left,right=span.split('-',1)
                            if left:start=int(left);end=min(int(right),end) if right else end
                            else:start=max(0,size-int(right))
                            if not 0<=start<=end<size:raise ValueError()
                        self.send_response(206 if range_header else 200)
                        self.send_header('Content-Type',kind);self.send_header('Accept-Ranges','bytes')
                        self.send_header('Content-Length',str(end-start+1))
                        if range_header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                        self.end_headers()
                        with open(path,'rb') as f:
                            f.seek(start);remaining=end-start+1
                            while remaining:
                                chunk=f.read(min(1024*1024,remaining))
                                if not chunk:break
                                self.wfile.write(chunk);remaining-=len(chunk)
                        return
                    data=path.read_bytes()
                self.send_response(200)
            except (KeyError,ValueError,IndexError,FileNotFoundError):
                data=b'Not found';kind='text/plain';self.send_response(404)
            except (BrokenPipeError,ConnectionResetError):return
            self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-cache');self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
    print(f'Track 1 review: http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
