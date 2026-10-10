// Compatibility values for the browser bundle; no host process or filesystem access.
import {Buffer} from 'buffer';
import process from 'process';
Object.assign(globalThis,{Buffer,process});
