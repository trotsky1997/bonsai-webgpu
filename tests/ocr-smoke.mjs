import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {createOcr} from '../web/ocr/ppocr.mjs';
const root=new URL('./fixtures/ocr/',import.meta.url);
const expected=JSON.parse(await readFile(new URL('expected.json',root),'utf8'));
const hashes=JSON.parse(await readFile(new URL('../web/ocr/SOURCE.json',import.meta.url),'utf8'));
const rows=[];
for(const [file,key]of [['ppocr_rs.wasm','compatible_sha256'],['ppocr_rs_fma.wasm','relaxed_sha256']]){
 const bytes=await readFile(new URL('../web/ocr/'+file,import.meta.url));assert.equal(createHash('sha256').update(bytes).digest('hex'),hashes[key]);
 const model=await createOcr({wasmBytes:bytes});
 for(const name of ['chinese.png','line-0.png','rotated.png','blank.png']){
  const result=model.ocr(await readFile(new URL(name,root)));
  if(expected.int8[name])assert.deepEqual(result.lines.map(l=>l.text),expected.int8[name].lines.map(l=>l.text));else assert(result.lines.some(l=>/[\u4e00-\u9fff]/.test(l.text)));
  rows.push({file,name,text:result.lines.map(l=>l.text),elapsed_ms:result.elapsed_ms});
 }
 assert.throws(()=>model.ocr(new Uint8Array([1,2,3])));model.dispose();assert.throws(()=>model.ocr(new Uint8Array()));
}
console.log(JSON.stringify(rows,null,2));
