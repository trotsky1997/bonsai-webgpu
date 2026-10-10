import type {Entry} from './runtime';
export function decode(entry?:Entry){return new TextDecoder().decode(entry?.content??new Uint8Array());}
const mime=(path:string)=>/\.png$/i.test(path)?'image/png':/\.jpe?g$/i.test(path)?'image/jpeg':/\.svg$/i.test(path)?'image/svg+xml':/\.webp$/i.test(path)?'image/webp':'application/octet-stream';
export function dataUrl(entry:Entry){let s='';const bytes=entry.content??new Uint8Array();for(let i=0;i<bytes.length;i+=8192)s+=String.fromCharCode(...bytes.subarray(i,i+8192));return 'data:'+mime(entry.path)+';base64,'+btoa(s);}
export function htmlPreview(file:Entry,entries:Entry[]){
 const template=document.createElement('template');template.innerHTML=decode(file);
 const asset=(name:string)=>{if(/^(?:[a-z]+:|\/\/|#)/i.test(name))return undefined;const path=new URL(name,'http://workspace'+file.path).pathname;return entries.find(e=>e.type==='file'&&e.path===path);};
 for(const node of template.content.querySelectorAll('base,meta[http-equiv="refresh" i],iframe,object,embed'))node.remove();
 for(const image of template.content.querySelectorAll('img')){const src=image.getAttribute('src')??'',entry=asset(src);if(entry)image.setAttribute('src',dataUrl(entry));else if(!src.startsWith('data:'))image.removeAttribute('src');image.removeAttribute('srcset');}
 for(const script of template.content.querySelectorAll('script[src]')){const entry=asset(script.getAttribute('src')??'');if(entry){script.removeAttribute('src');script.textContent=decode(entry).replace(/<\/script/gi,'<\\/script');}else script.remove();}
 for(const link of template.content.querySelectorAll('link')){const entry=asset(link.getAttribute('href')??'');if(entry&&link.getAttribute('rel')==='stylesheet'){const style=document.createElement('style');style.textContent=decode(entry);link.replaceWith(style);}else link.remove();}
 const policy="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; worker-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'";
 return '<!doctype html><meta http-equiv="Content-Security-Policy" content="'+policy+'"><meta name="viewport" content="width=device-width, initial-scale=1">'+template.innerHTML;
}
