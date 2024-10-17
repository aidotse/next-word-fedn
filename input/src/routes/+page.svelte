<script lang="ts">
	import { onMount } from 'svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '$lib/components/ui/card';
  
	let messages: any[] = [];
	let newMessage = '';
	let autocompleteText = '';
	let selectedModel = 'GRU';
  
	let models = ['LSTM', 'GRU'];
  
	onMount(() => {
	  messages = [
		{ id: 1, text: 'Hello!', sender: 'other' },
		{ id: 2, text: 'Hi there!', sender: 'self' }
	  ];
	});
  
	async function handleInput(event: Event) {
	  const inputElement = event.target as HTMLTextAreaElement;
	  const lastChar = inputElement.value.slice(-1);
	  
	  if (lastChar === ' ' && newMessage.trim()) {
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
  
	function handleSubmit() {
	  if (newMessage.trim()) {
		messages = [...messages, { id: messages.length + 1, text: newMessage, sender: 'self' }];
		newMessage = '';
		autocompleteText = '';
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
  </script>
  
  <style>
	.sidebar {
	  width: 250px;
	  background-color: #f8f9fa;
	  padding: 20px;
	  height: 100vh;
	  position: fixed;
	  left: 0;
	  top: 0;
	  bottom: 0;
	  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
	  overflow-y: auto;
	}
  
	.content {
	  margin-left: 270px;
	  padding: 20px;
	  max-width: 800px;
	  margin: 0 auto;
	}

	.message-container {
	  max-height: 400px;
	  overflow-y: auto;
	  padding: 10px;
	  border: 1px solid #e0e0e0;
	  border-radius: 5px;
	  margin-bottom: 20px;
	}

	.message {
	  padding: 10px;
	  margin-bottom: 10px;
	  border-radius: 10px;
	  max-width: 80%;
	}

	.message.self {
	  background-color: #e3f2fd;
	  margin-left: auto;
	}

	.message.other {
	  background-color: #f5f5f5;
	}

	.input-container {
	  position: relative;
	  margin-bottom: 20px;
	}

	.autocomplete {
	  position: absolute;
	  bottom: 5px;
	  left: 10px;
	  color: #9e9e9e;
	  pointer-events: none;
	}
  </style>
  
  <div class="sidebar">
	<h2 class="font-bold text-xl mb-6">Select Model</h2>
	<div class="flex flex-col space-y-3">
	  {#each models as model}
		<Button
		  on:click={() => selectModel(model)}
		  class={`w-full transition-colors duration-200 ${selectedModel === model ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-200 hover:bg-gray-300'}`}
		>
		  {model}
		</Button>
	  {/each}
	</div>
  </div>
  
  <div class="content">
	<Card class="w-full shadow-lg">
	  <CardHeader>
		<CardTitle class="text-2xl font-bold text-center">Autocomplete Chat</CardTitle>
	  </CardHeader>
	  <CardContent>
		<div class="message-container">
		  {#each messages as message (message.id)}
			<div
			  class={`message ${message.sender === 'self' ? 'self' : 'other'}`}
			>
			  {message.text}
			</div>
		  {/each}
		</div>
		<div class="input-container">
		  <Textarea
			bind:value={newMessage}
			on:input={handleInput}
			placeholder="Type your message here..."
			class="w-full min-h-[100px] p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
		  />
		  {#if autocompleteText && newMessage.trim()}
			<div class="autocomplete">
			  {newMessage}{autocompleteText}
			</div>
		  {/if}
		</div>
	  </CardContent>
	  <CardFooter class="flex justify-end">
		<Button on:click={handleSubmit} class="bg-blue-600 text-white hover:bg-blue-700 transition-colors duration-200">Send</Button>
	  </CardFooter>
	</Card>
  </div>