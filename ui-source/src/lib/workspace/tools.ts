import type {OpenAIToolDefinition} from '$lib/types';
export const WORKSPACE_TOOLS=['bash','read','write','edit','grep','glob'];
const str={type:'string'};
const tool=(name:string,description:string,properties:Record<string,unknown>,required:string[]):OpenAIToolDefinition=>({type:'function',function:{name,description,parameters:{type:'object',properties,required}}});
export const workspaceTools=()=>[
 tool('bash','Run shell commands in the persistent virtual /workspace. Supports ls, cat, grep, sed, awk, jq, pipes and redirects. No network, Python, Node, npm or native binaries.',{command:str},['command']),
 tool('read','Read a workspace UTF-8 file. Inspect files before editing.',{path:str,offset:{type:'integer',description:'Starting line, 1-based'},limit:{type:'integer',description:'Maximum lines, default 100'}},['path']),
 tool('write','Create or replace a UTF-8 file. HTML files appear in the artifact panel. Use self-contained HTML, inline CSS/JS, no CDN.',{path:str,content:str},['path','content']),
 tool('glob','Find virtual workspace files by glob pattern (*, **, ?, character classes), e.g. **/*.html. Returns sorted paths. No host filesystem.',{pattern:str,path:{type:'string',description:'Base directory, default /workspace'},limit:{type:'integer',description:'Maximum paths, default 200, max 1000'}},['pattern']),
 tool('grep','Search virtual file contents with a regular expression (or literal=true). Returns path:line:match. Recursive under path; optional glob filters files.',{pattern:str,path:{type:'string',description:'File or directory, default /workspace'},glob:{type:'string'},ignore_case:{type:'boolean'},literal:{type:'boolean'},limit:{type:'integer',description:'Maximum matching lines, default 100, max 200'}},['pattern']),
 tool('edit','Replace exactly one occurrence in a file. Read the file first.',{path:str,oldText:str,newText:str},['path','oldText','newText'])
];
