<template>
  <div class="flex justify-center items-center w-full h-full" v-html="svg" />
</template>

<script setup>
import { renderMermaidSVG, THEMES } from 'beautiful-mermaid'
import { computed } from 'vue'

const props = defineProps({
  diagram: { type: String, required: true },
  theme:   { type: String, default: 'zinc-light' },
  options: { type: Object, default: () => ({}) },
})

const svg = computed(() =>
  renderMermaidSVG(props.diagram, { ...THEMES[props.theme], ...props.options })
)
</script>
