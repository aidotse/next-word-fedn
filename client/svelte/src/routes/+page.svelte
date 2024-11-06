<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		Card,
		CardContent,
		CardFooter,
		CardHeader,
		CardTitle
	} from '$lib/components/ui/card/index.js';

	import { ModeWatcher } from 'mode-watcher';

	let newMessage = '';
	let autocompleteText = '';
	let topwords: any[] = [];
	let textareaElement: HTMLTextAreaElement;
	let displaytext = '';
	let isUpdating = false;
	let updateError = '';

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
						num_words: 1
					})
				});
				const data = await response.json();
				autocompleteText = data.generated_text.split(' ').pop();
				topwords = data.top_3;
				console.log(topwords);
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
			newMessage += autocompleteText + ' ';
			await fetchAutocomplete();
			await tick();
			scrollTextareaToBottom();
		}
	}

	function scrollTextareaToBottom() {
		if (textareaElement) {
			textareaElement.scrollTop = textareaElement.scrollHeight;
		}
	}

	async function updatemodel() {
		isUpdating = true;
		updateError = '';
		try {
			const response = await fetch('http://localhost:5000/update-model', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				}
			});
			const data = await response.json();
			if (data.success) {
				displaytext = data.message;
			} else {
				updateError = data.message;
			}
		} catch (error) {
			console.error('Error updating model:', error);
			updateError = 'Failed to update model. Please try again.';
		} finally {
			isUpdating = false;
			setTimeout(() => {
				displaytext = '';
				updateError = '';
			}, 3000);
		}
	}
</script>

<button
	class="absolute top-4 left-4 px-4 py-2 bg-blue-700 hover:bg-blue-800 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
	on:click={updatemodel}
	disabled={isUpdating}
>
	{#if isUpdating}
		<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
	{/if}
	{isUpdating ? 'Updating...' : 'Update Model'}
</button>

{#if displaytext}
	<div
		class="absolute top-4 left-1/2 transform -translate-x-1/2 bg-green-100 dark:bg-green-800 px-4 py-2 rounded-lg text-green-700 dark:text-green-200"
	>
		{displaytext}
	</div>
{/if}

{#if updateError}
	<div
		class="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-100 dark:bg-red-800 px-4 py-2 rounded-lg text-red-700 dark:text-red-200"
	>
		{updateError}
	</div>
{/if}

<div class="container relative">
	<Card class="w-full shadow-lg mt-24">
		<CardHeader>
			<CardTitle class="text-2xl font-bold text-center">Text Autocomplete</CardTitle>
		</CardHeader>
		<CardContent>
			<div class="relative">
				{#if autocompleteText && newMessage.trim()}
					<div class="absolute bottom-full left-0 mb-2 flex gap-2">
						{#each topwords as suggestion, index}
							<button
								class={`px-3 py-1 rounded-lg text-sm transition-colors ${
									index === 0
										? 'bg-blue-700 hover:bg-blue-800 text-white'
										: 'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600'
								}`}
								on:click={() => {
									newMessage += suggestion;
									textareaElement.focus();
								}}
							>
								{suggestion}
							</button>
						{/each}
					</div>
				{/if}
				<textarea
					bind:this={textareaElement}
					bind:value={newMessage}
					on:input={handleInput}
					on:keydown={handleKeydown}
					placeholder="Start typing..."
					class="w-full min-h-[200px] p-3 border rounded-lg focus:outline-none focus:ring-0 relative z-10 bg-transparent font-sans text-base leading-normal resize-y"
				></textarea>
			</div>
		</CardContent>
	</Card>
</div>

<slot />
