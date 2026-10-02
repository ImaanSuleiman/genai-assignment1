import React, { useState } from 'react'
import { Card, ErrorBox, fmtMs, ImagePanel, ImageSource, RunButton, Segmented, sourceFields, Stat, useInference } from '../components/common'

const STYLES = [1, 2, 3].map((n) => ({ value: n, label: `Style ${n}` }))

export default function FaceToSketch() {
  const [src, setSrc] = useState(null)
  const [style, setStyle] = useState(1)
  const { loading, error, result, run } = useInference('face2sketch')

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold">Face-to-Sketch Generator</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          A style-conditioned conditional GAN (U-Net generator, PatchGAN discriminator) turns a face photograph into a sketch in one of three FS2K styles.
        </p>
      </div>
      <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
        <Card title="Photo">
          <div className="space-y-4">
            <ImageSource value={src} onChange={setSrc} samples={false} webcam />
            <Segmented label="Sketch style" value={style} onChange={setStyle} options={STYLES} />
            <RunButton loading={loading} disabled={!src} onClick={() => run({ ...sourceFields(src), style })}>Generate sketch</RunButton>
            <ErrorBox message={error} />
          </div>
        </Card>
        <Card title="Result" subtitle={result ? `${result.style} · source: ${result.source.source}` : 'Upload or capture a face photo, pick a style and generate.'}>
          <div className="grid grid-cols-2 gap-5">
            <ImagePanel title="Original photograph" src={result?.photo_image} />
            <ImagePanel title={`Generated sketch (${result?.style ?? 'Style ' + style})`} src={result?.sketch_image} download={`sketch_style${style}.png`} />
          </div>
          {result && (
            <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3">
              <Stat label="Inference time" value={fmtMs(result.inference_ms)} />
              <Stat label="Model" value={result.model} />
              <Stat label="Output size" value="128 × 128" />
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
