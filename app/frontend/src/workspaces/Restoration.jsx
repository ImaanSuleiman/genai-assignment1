import React, { useState } from 'react'
import {
  Bar, Card, CORRUPTIONS, ErrorBox, fmtMs, ImagePanel, ImageSource, LEVELS, RunButton, Segmented, sourceFields, Stat, useInference,
} from '../components/common'

/** Shared layout for the three restoration workspaces (Tasks 1-3). `variant` selects the endpoint + extra output. */
const CONFIG = {
  universal: {
    endpoint: 'universal', title: 'Universal Restoration',
    blurb: 'One denoising autoencoder restores clean, salt-and-pepper, blurred and occluded images without being told which corruption was applied.',
  },
  hard: {
    endpoint: 'hard-route', title: 'Hard-Routed Restoration',
    blurb: 'A classifier predicts the corruption, then exactly one specialist autoencoder restores the image. Clean inputs use an identity bypass.',
  },
  soft: {
    endpoint: 'soft-moe', title: 'Soft Mixture-of-Experts Restoration',
    blurb: 'A gate assigns a continuous weight to every branch (identity + three experts) and the output is their weighted sum.',
  },
}

export default function Restoration({ variant }) {
  const cfg = CONFIG[variant]
  const [src, setSrc] = useState(null)
  const [corruption, setCorruption] = useState('salt')
  const [second, setSecond] = useState('none')
  const [level, setLevel] = useState('medium')
  const [seed, setSeed] = useState(42)
  const { loading, error, result, run } = useInference(cfg.endpoint)
  const applied = corruption !== 'already_corrupted' && corruption !== 'none'

  const submit = () =>
    run({ ...sourceFields(src), corruption, level, seed, ...(variant === 'soft' ? { second_corruption: second } : {}) })

  const settingsText = result
    ? Object.entries(result.settings).map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : Array.isArray(v) ? JSON.stringify(v) : v}`).join('  ·  ')
    : ''

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold">{cfg.title}</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">{cfg.blurb}</p>
      </div>
      <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
        <Card title="Input">
          <div className="space-y-4">
            <ImageSource value={src} onChange={setSrc} samples />
            <Segmented label="Corruption" value={corruption} onChange={setCorruption} options={CORRUPTIONS} />
            {applied && <Segmented label="Severity (fixed test levels)" value={level} onChange={setLevel} options={LEVELS} />}
            {variant === 'soft' && applied && (
              <Segmented label="Add a second corruption (stress test)" value={second} onChange={setSecond}
                options={[{ value: 'none', label: 'None' }, ...CORRUPTIONS.filter((c) => ['salt', 'blur', 'occlusion'].includes(c.value) && c.value !== corruption)]} />
            )}
            {applied && (
              <label className="block text-xs font-medium text-slate-500">Random seed
                <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} className="ml-2 w-24 rounded-lg border border-slate-200 px-2 py-1 text-sm" />
              </label>
            )}
            <RunButton loading={loading} disabled={!src} onClick={submit}>Restore image</RunButton>
            <ErrorBox message={error} />
          </div>
        </Card>

        <div className="space-y-5">
          <Card title="Result" subtitle={result ? settingsText : 'Choose an image and press “Restore image”.'}>
            <div className={`grid gap-4 ${result?.clean_reference ? 'grid-cols-2 md:grid-cols-4' : 'grid-cols-3'}`}>
              {result?.clean_reference && <ImagePanel title="Clean reference" src={result.clean_reference} />}
              <ImagePanel title="Input (corrupted)" src={result?.input_image} />
              <ImagePanel title="Restored output" src={result?.restored_image} download="restored.png" />
              <ImagePanel title="Absolute error" src={result?.error_map} caption="brighter = larger error" />
            </div>
            {result && (
              <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Inference time" value={fmtMs(result.inference_ms)} />
                {result.metrics && <Stat label="PSNR input → output" value={`${result.metrics.psnr_input.toFixed(1)} → ${result.metrics.psnr_restored.toFixed(1)} dB`} />}
                {variant === 'hard' && <Stat label="Classifier / expert" value={`${fmtMs(result.classifier_ms)} / ${fmtMs(result.expert_ms)}`} />}
                <Stat label="Image source" value={result.source.source} />
              </div>
            )}
          </Card>

          {variant === 'hard' && result && (
            <Card title="Classifier probabilities" subtitle={`Predicted: ${result.predicted_corruption} → ${result.selected_expert}${result.classifier_correct === false ? '  (misrouted)' : ''}`}>
              <div className="grid gap-3 md:grid-cols-2">
                {Object.entries(result.probabilities).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} highlight={k === result.predicted_corruption} />
                ))}
              </div>
              {result.identity_bypass && <p className="mt-3 text-xs text-slate-500">Predicted clean: identity bypass, no restoration expert was run.</p>}
            </Card>
          )}

          {variant === 'soft' && result && (
            <Card title="Routing weights" subtitle={`Strongest contributor: ${result.dominant_expert}`}>
              <div className="grid gap-3 md:grid-cols-2">
                {Object.entries(result.weights).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} highlight={k === result.dominant_expert} />
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
