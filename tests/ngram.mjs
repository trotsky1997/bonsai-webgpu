import assert from 'node:assert/strict';
import {NgramDraft} from '../web/ngram.mjs';
const n=new NgramDraft([1,2,3,4,5,6,7,8,9,1,2,3]);assert.deepEqual(n.next(4),[4,5,6,7]);
n.append(99);assert.deepEqual(n.next(4),[]);assert.deepEqual(new NgramDraft([1,2]).next(4),[]);
for(let i=0;i<200;i++)n.append(i%9);for(const positions of n.index.values())assert(positions.length<=4);
assert(n.next(7).length<=7);assert.deepEqual(n.next(0),[]);
console.log('N-gram bounded lookup tests passed');
