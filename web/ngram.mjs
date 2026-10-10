// Deterministic lookup drafts. Only actual prompt/output tokens enter the index.
export class NgramDraft {
 constructor(tokens=[],min=3,max=8){this.tokens=[];this.index=new Map();this.min=min;this.max=max;for(const t of tokens)this.append(t);}
 append(token){const n=this.tokens.length;for(let k=this.min;k<=Math.min(this.max,n);k++){const key=this.tokens.slice(n-k).join(','),positions=this.index.get(key)??[];positions.push(n);if(positions.length>4)positions.shift();this.index.set(key,positions);}this.tokens.push(token);}
 next(limit){const n=this.tokens.length;for(let k=Math.min(this.max,n);k>=this.min;k--){const positions=this.index.get(this.tokens.slice(n-k).join(','));if(positions?.length){const at=positions.find(p=>n-p>=limit)??positions[0];return this.tokens.slice(at,Math.min(n,at+limit));}}return [];}
}
