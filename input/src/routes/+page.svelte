<script lang="ts">
	import { onMount } from 'svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '$lib/components/ui/card';

	let messages: any[] = [];
	let newMessage = '';

	onMount(() => {
		// Here you would typically fetch initial messages from a server
		messages = [
			{ id: 1, text: 'Hello!', sender: 'other' },
			{ id: 2, text: 'Hi there!', sender: 'self' }
		];
	});

	function handleInput() {
		// You can add real-time features here, like "user is typing" indicators
	}

	function handleSubmit() {
		if (newMessage.trim()) {
			messages = [...messages, { id: messages.length + 1, text: newMessage, sender: 'self' }];
			newMessage = '';
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
			<Textarea
				bind:value={newMessage}
				on:input={handleInput}
				placeholder="Type your message here..."
				class="w-full min-h-[50px]"
			/>
		</CardContent>
		<CardFooter class="flex justify-end">
			<Button on:click={handleSubmit}>Send</Button>
		</CardFooter>
	</Card>
</main>
