import {gzipSync as compress,gunzipSync as expand} from 'fflate';
import {Buffer} from 'buffer';
export const constants={Z_BEST_COMPRESSION:9,Z_BEST_SPEED:1,Z_DEFAULT_COMPRESSION:-1};
export function gunzipSync(input:Uint8Array){return Buffer.from(expand(input));}
export function gzipSync(input:Uint8Array,options?:{level?:number;mtime?:number}){const level=options?.level;return Buffer.from(compress(input,{level:(level===undefined||level<0?6:level) as 0|1|2|3|4|5|6|7|8|9,mtime:options?.mtime??0}));}
