<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Button } from "$lib/components/ui/button";
	import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';

	import { ModeWatcher } from "mode-watcher";

	import Sun from "lucide-svelte/icons/sun";
	import Moon from "lucide-svelte/icons/moon";
   
	import { toggleMode } from "mode-watcher";

	let messages: any[] = [];
	let newMessage = '';
	let autocompleteText = '';
	let selectedModel = 'GRU';
  
	let models = ['LSTM', 'GRU'];
	
	let messageContainer: HTMLElement;
	let textareaElement: HTMLTextAreaElement;
  
	onMount(() => {
	  messages = [
		{ id: 1, text: 'Hello!', sender: 'other' },
		{ id: 2, text: 'Hi there!', sender: 'self' }
	  ];
	  scrollToBottom();
	});
  
	async function fetchAutocomplete() {
	  if (newMessage.trim()) {
		try {
		  const response = await fetch('http://localhost:5000/generate', {
			method: 'POST',
			headers: {
			  'Content-Type': 'application/json'
			},
			body: JSON.stringify({
			  seed_text: newMessage.trim(),
			  num_words: 1,
			  model: selectedModel
			})
		  });
		  const data = await response.json();
		  autocompleteText = data.generated_text.split(' ').pop();
		} catch (error) {
		  console.error('Error fetching autocomplete:', error);
		}
	  } else {
		autocompleteText = '';
	  }
	}

	async function handleInput(event: Event) {
	  const inputElement = event.target as HTMLTextAreaElement;
	  const lastChar = inputElement.value.slice(-1);
	  
	  if (lastChar === ' ') {
		await fetchAutocomplete();
	  } else {
		autocompleteText = '';
	  }
	  await tick();
	  scrollTextareaToBottom();
	}
  
	async function handleKeydown(event: KeyboardEvent) {
	  if (event.key === 'Tab' && autocompleteText) {
		event.preventDefault();
		newMessage += autocompleteText+' ';
		await fetchAutocomplete();
		await tick();
		scrollTextareaToBottom();
	  } else if (event.key === 'Enter' && !event.shiftKey) {
		event.preventDefault();
		handleSubmit();
	  }
	}
  
	function handleSubmit() {
	  if (newMessage.trim()) {
		messages = [...messages, { id: messages.length + 1, text: newMessage, sender: 'self' }];
		newMessage = '';
		autocompleteText = '';
		scrollToBottom();
	  }
	}
  
	function selectModel(model: string) {
	  selectedModel = model;
	  if (selectedModel === 'LSTM') {
		fetch('http://localhost:5000/model-type', {
		  method: 'POST',
		  headers: {
			'Content-Type': 'application/json'
		  },
		  body: JSON.stringify({ model_type: 'lstm_onehot' })
		});
	  } else if (selectedModel === 'GRU') {
		fetch('http://localhost:5000/model-type', {
		  method: 'POST',
		  headers: {
			'Content-Type': 'application/json'
		  },
		  body: JSON.stringify({ model_type: 'gru_onehot' })
		});
	  }
	}

	function scrollToBottom() {
	  setTimeout(() => {
		if (messageContainer) {
		  messageContainer.scrollTop = messageContainer.scrollHeight;
		}
	  }, 0);
	}

	function scrollTextareaToBottom() {
	  if (textareaElement) {
		textareaElement.scrollTop = textareaElement.scrollHeight;
	  }
	}
  </script>
  
  <div class="fixed top-5 right-5 z-50">
	<Button on:click={toggleMode} variant="outline" size="icon">
	  <Sun
		class="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0"
	  />
	  <Moon
		class="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100"
	  />
	  <span class="sr-only">Toggle theme</span>
	</Button>
  </div>

  <div class="w-64 p-5 h-screen fixed left-0 top-0 bottom-0 shadow-md overflow-y-auto transition-colors duration-300 bg-gray-100 dark:bg-gray-800 text-black dark:text-white">
	<h2 class="font-bold text-xl mb-6">Select Model</h2>
	<div class="flex flex-col space-y-3">
	  {#each models as model}
		<Button
		  on:click={() => selectModel(model)}
		  class={`w-full transition-colors duration-200 ${selectedModel === model ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600'}`}
		>
		  {model}
		</Button>
	  {/each}
	</div>
  </div>
  
  <div class="ml-64 p-5 max-w-3xl mx-auto">
	<Card class="w-full shadow-lg">
	  <CardHeader>
		<CardTitle class="text-2xl font-bold text-center">Autocomplete Chat</CardTitle>
	  </CardHeader>
	  <CardContent>
		<div bind:this={messageContainer} class="max-h-96 overflow-y-auto p-2.5 border border-gray-200 dark:border-gray-700 rounded-lg mb-5">
		  {#each messages as message (message.id)}
			<div
			  class={`p-2.5 mb-2.5 rounded-lg max-w-[80%] ${message.sender === 'self' ? 'bg-blue-100 dark:bg-blue-900 ml-auto' : 'bg-gray-100 dark:bg-gray-700'}`}
			>
			  {message.text}
			</div>
		  {/each}
		</div>
		<div class="relative mb-5">
		  <textarea
			bind:this={textareaElement}
			bind:value={newMessage}
			on:input={handleInput}
			on:keydown={handleKeydown}
			placeholder="Type your message here..."
			class="w-full min-h-[50px] p-3 border rounded-lg focus:outline-none focus:ring-0 relative z-10 bg-transparent font-sans text-base leading-normal resize-y"
		  ></textarea>
		  {#if autocompleteText && newMessage.trim()}
			<div class="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
			  <div
				class="w-full min-h-[50px] p-3 text-gray-500 whitespace-pre-wrap break-words absolute top-0 left-0 font-sans text-base leading-normal"
			  >
				<span class="invisible">{newMessage}</span><span class="visible">{autocompleteText}</span>
			  </div>
			</div>
		  {/if}	
		</div>
	  </CardContent>
	  <CardFooter class="flex justify-end">
		<Button on:click={handleSubmit} class="bg-blue-600 text-white hover:bg-blue-700 transition-colors duration-200">Send</Button>
	  </CardFooter>
	</Card>
  </div>

  <ModeWatcher />
<slot />