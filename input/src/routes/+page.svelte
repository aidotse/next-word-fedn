<script lang="ts">
	import { onMount } from 'svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '$lib/components/ui/card';

	let messages: any[] = [];
	let newMessage = '';
	let autocompleteText = '';

	onMount(() => {
		// Here you would typically fetch initial messages from a server
		messages = [
			{ id: 1, text: 'Hello!', sender: 'other' },
			{ id: 2, text: 'Hi there!', sender: 'self' }
		];
	});

	async function handleInput() {
		if (newMessage.trim()) {
			try {
				const response = await fetch('http://10.132.177.3:5000/generate', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						seed_text: newMessage,
						num_words: 1
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
			// Here you would typically send the message to a server
		}
	}
</script>

<main class="container mx-auto p-4">
	<Card class="w-full max-w-md mx-auto">
		<CardHeader>
			<CardTitle>Autocomplete Test</CardTitle>
		</CardHeader>
		<CardContent>
			<div class="mb-4 max-h-60 overflow-y-auto">
				{#each messages as message (message.id)}
					<div
						class={`mb-2 p-2 rounded ${message.sender === 'self' ? 'bg-blue-100 ml-auto' : 'bg-gray-100'}`}
					>
						{message.text}
					</div>
				{/each}
			</div>
			<div class="relative">
				<Textarea
					bind:value={newMessage}
					on:input={handleInput}
					placeholder="Type your message here..."
					class="w-full min-h-[50px]"
				/>
				{#if autocompleteText}
					<div class="absolute bottom-2 left-2 text-gray-400">
						{newMessage}{autocompleteText}
					</div>
				{/if}
			</div>
		</CardContent>
		<CardFooter class="flex justify-end">
			<Button on:click={handleSubmit}>Send</Button>
		</CardFooter>
	</Card>
</main>
