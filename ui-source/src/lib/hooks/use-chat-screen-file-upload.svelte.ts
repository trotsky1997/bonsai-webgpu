/**
 * File upload lifecycle for the ChatScreen form.
 *
 * Owns the queue of processed `ChatUploadedFile`, the rejection-by-capability
 * dialog state, and the dual-layer validation pipeline (general format +
 * model modality). The caller provides the active model's capabilities and ID
 * as reactive getters so validation tracks the model in real time.
 */

import { onDestroy } from 'svelte';
import { recognizeImage } from '$lib/ocr/client';
import { filterFilesByModalities, isFileTypeSupported } from '$lib/utils';
import { processFilesToChatUploaded } from '$lib/utils/browser-only';

interface UseChatScreenFileUploadOptions {
	capabilities: () => { hasVision: boolean; hasAudio: boolean; hasVideo: boolean };
	activeModelId: () => string | null | undefined;
}

export interface FileErrorData {
	generallyUnsupported: File[];
	modalityUnsupported: File[];
	modalityReasons: Record<string, string>;
	supportedTypes: string[];
}

const completedImages=new Map<string,Partial<ChatUploadedFile>>();
const remember=(id:string,patch:Partial<ChatUploadedFile>)=>{completedImages.set(id,patch);if(completedImages.size>128)completedImages.delete(completedImages.keys().next().value!);};

export function useChatScreenFileUpload(options: UseChatScreenFileUploadOptions) {
	const pendingImages = new Map<string, AbortController>();
	onDestroy(() => { for (const [id,controller] of pendingImages){remember(id,{isLoading:false,loadError:'图片处理已中断，请重新上传'});controller.abort();} });
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let showFileErrorDialog = $state(false);
	let fileErrorData = $state<FileErrorData>({
		generallyUnsupported: [],
		modalityReasons: {},
		modalityUnsupported: [],
		supportedTypes: []
	});

	async function processFiles(files: File[]) {
		const generallySupported: File[] = [];
		const generallyUnsupported: File[] = [];

		for (const file of files) {
			if (isFileTypeSupported(file.name, file.type)) {
				generallySupported.push(file);
			} else {
				generallyUnsupported.push(file);
			}
		}

		const { modalityReasons, supportedFiles, unsupportedFiles } = filterFilesByModalities(
			generallySupported,
			{ ...options.capabilities(), hasVision: true }
		);
		const allUnsupportedFiles = [...generallyUnsupported, ...unsupportedFiles];

		if (allUnsupportedFiles.length > 0) {
			const supportedTypes: string[] = ['text files', 'PDFs'];
			const caps = options.capabilities();

			if (caps.hasVision) supportedTypes.push('images');

			if (caps.hasAudio) supportedTypes.push('audio files');

			if (caps.hasVideo) supportedTypes.push('video files');

			fileErrorData = {
				generallyUnsupported,
				modalityReasons,
				modalityUnsupported: unsupportedFiles,
				supportedTypes
			};
			showFileErrorDialog = true;
		}

		if (supportedFiles.length > 0) {
			const processed = await processFilesToChatUploaded(
				supportedFiles,
				options.activeModelId() ?? undefined
			);

			const entries=processed.map(file => file.type.startsWith('image/') ? {...file,isLoading:true} : file);
			uploadedFiles = [...uploadedFiles, ...entries];
			for(const file of entries){
				if(!file.type.startsWith('image/'))continue;
				const controller=new AbortController();pendingImages.set(file.id,controller);
				void (async()=>{
					try{
						if(file.size>20*1024*1024)throw Error('图片超过 20 MB，请缩小后重试');
						if(!file.preview)throw Error('图片格式无法解码');
						const imageText=await recognizeImage(file.preview,controller.signal);
						remember(file.id,{imageText,isLoading:false});uploadedFiles=uploadedFiles.map(f=>f.id===file.id?{...f,imageText,isLoading:false}:f);
					}catch(e){if(!controller.signal.aborted){const patch={isLoading:false,loadError:String((e as Error)?.message??e)};remember(file.id,patch);uploadedFiles=uploadedFiles.map(f=>f.id===file.id?{...f,...patch}:f);}}
					finally{pendingImages.delete(file.id);}
				})();
			}
		}
	}

	function handleFileUpload(files: File[]) {
		return processFiles(files);
	}

	function handleFileRemove(fileId: string) {
		pendingImages.get(fileId)?.abort();
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
	}

	return {
		get fileErrorData() {
			return fileErrorData;
		},
		handleFileRemove,
		handleFileUpload,
		get showFileErrorDialog() {
			return showFileErrorDialog;
		},
		set showFileErrorDialog(value) {
			showFileErrorDialog = value;
		},
		get uploadedFiles() {
			return uploadedFiles;
		},
		set uploadedFiles(value) {
			uploadedFiles = value.map(f=>({...f,...(completedImages.get(f.id)??(f.isLoading&&!pendingImages.has(f.id)?{isLoading:false,loadError:'图片处理已中断，请重新上传'}:{}))}));
		}
	};
}
