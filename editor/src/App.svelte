<script lang="ts">
  type BBox = [number, number, number, number];
  type Candidate = { status: string; bbox?: BBox };
  type Format = { name: string; baseline: Candidate; enhanced: Candidate };
  type Person = { person_id: number; confidence: number; bbox: BBox };
  type Face = { face_id: number; bbox: BBox };
  type Sample = {
    id: string;
    original_path: string;
    source: string;
    image_size: [number, number];
    persons: Person[];
    faces: Face[];
    formats: Format[];
  };
  type Decision =
    "baseline" | "enhanced" | "manual" | "source_bound" | "skipped";
  type Review = {
    sample_id: string;
    format: string;
    decision: Decision;
    expected_bbox?: BBox;
    reason_codes: string[];
    note: string;
  };

  const reasonOptions = [
    ["face_cut", "Face cut"],
    ["face_edge_tight", "Face edge-tight"],
    ["subject_too_high", "Subject too high"],
    ["subject_too_low", "Subject too low"],
    ["subject_too_left", "Subject too left"],
    ["subject_too_right", "Subject too right"],
    ["wrong_subject", "Wrong subject"],
    ["group_framing", "Group framing"],
    ["detector_error", "Detector error"],
  ] as const;

  let sampleIds: string[] = [];
  let samples = new Map<string, Sample>();
  let selectedIndex = 0;
  let selectedFormat = "21:9";
  let manualCrop: BBox | undefined;
  let reviews = new Map<string, Review>();
  let drafts = new Map<string, Pick<Review, "reason_codes" | "note">>();
  let showPersons = true;
  let showFaces = true;
  let loading = true;
  let error = "";
  let dragging = false;
  let dragOffset: [number, number] = [0, 0];
  let canvas: HTMLDivElement;

  $: selectedId = sampleIds[selectedIndex];
  $: sample = selectedId ? samples.get(selectedId) : undefined;
  $: format = sample?.formats.find((item) => item.name === selectedFormat);
  $: baseline = format?.baseline.bbox;
  $: enhanced = format?.enhanced.bbox;
  $: activeReview = selectedId
    ? reviews.get(reviewKey(selectedId, selectedFormat))
    : undefined;
  $: activeDraft = selectedId
    ? drafts.get(reviewKey(selectedId, selectedFormat))
    : undefined;
  $: if (sample && format) resetCrop();

  loadManifest();

  async function loadManifest() {
    try {
      const manifest = await fetch("/data/manifest.json").then(requireOk);
      sampleIds = manifest.samples;
      await Promise.all(sampleIds.map(loadSample));
    } catch (cause) {
      error =
        cause instanceof Error ? cause.message : "Could not load editor/data.";
    } finally {
      loading = false;
    }
  }

  async function loadSample(id: string) {
    const response = await fetch(`/data/samples/${id}.json`).then(requireOk);
    samples.set(id, response as Sample);
    samples = samples;
  }

  async function requireOk(response: Response) {
    if (!response.ok)
      throw new Error("Run `icarus-v2 --review --input <path>` first.");
    return response.json();
  }

  function resetCrop() {
    manualCrop = baseline ? ([...baseline] as BBox) : undefined;
  }

  function chooseCandidate(kind: "baseline" | "enhanced") {
    const candidate = kind === "baseline" ? baseline : enhanced;
    if (!candidate || !sample) return;
    manualCrop = [...candidate] as BBox;
    saveReview(kind, candidate);
  }

  function saveManual() {
    if (manualCrop) saveReview("manual", manualCrop);
  }

  function saveReview(decision: Decision, expected_bbox?: BBox) {
    if (!sample) return;
    reviews.set(reviewKey(sample.id, selectedFormat), {
      sample_id: sample.id,
      format: selectedFormat,
      decision,
      expected_bbox: expected_bbox ? ([...expected_bbox] as BBox) : undefined,
      reason_codes:
        activeDraft?.reason_codes ?? activeReview?.reason_codes ?? [],
      note: activeDraft?.note ?? activeReview?.note ?? "",
    });
    reviews = reviews;
  }

  function mark(decision: "source_bound" | "skipped") {
    saveReview(decision);
  }

  function reviewKey(id: string, format: string) {
    return `${id}\u001f${format}`;
  }

  function toggleReason(code: string) {
    if (!sample) return;
    const draft = activeDraft ?? {
      reason_codes: activeReview?.reason_codes ?? [],
      note: activeReview?.note ?? "",
    };
    const reason_codes = draft.reason_codes.includes(code)
      ? draft.reason_codes.filter((value) => value !== code)
      : [...draft.reason_codes, code];
    drafts.set(reviewKey(sample.id, selectedFormat), {
      ...draft,
      reason_codes,
    });
    drafts = drafts;
  }

  function updateNote(event: Event) {
    if (!sample) return;
    const draft = activeDraft ?? {
      reason_codes: activeReview?.reason_codes ?? [],
      note: activeReview?.note ?? "",
    };
    drafts.set(reviewKey(sample.id, selectedFormat), {
      ...draft,
      note: (event.target as HTMLTextAreaElement).value,
    });
    drafts = drafts;
  }

  function beginDrag(event: PointerEvent) {
    if (!manualCrop || !sample || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = sample.image_size[0] / rect.width;
    const scaleY = sample.image_size[1] / rect.height;
    dragOffset = [
      event.clientX - rect.left - manualCrop[0] / scaleX,
      event.clientY - rect.top - manualCrop[1] / scaleY,
    ];
    dragging = true;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  }

  function dragCrop(event: PointerEvent) {
    if (!dragging || !manualCrop || !sample || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = sample.image_size[0] / rect.width;
    const scaleY = sample.image_size[1] / rect.height;
    const width = manualCrop[2] - manualCrop[0];
    const height = manualCrop[3] - manualCrop[1];
    const x = clamp(
      (event.clientX - rect.left - dragOffset[0]) * scaleX,
      0,
      sample.image_size[0] - width,
    );
    const y = clamp(
      (event.clientY - rect.top - dragOffset[1]) * scaleY,
      0,
      sample.image_size[1] - height,
    );
    const adjusted = [x, y, x + width, y + height] as BBox;
    manualCrop = adjusted;
    if (sample && activeReview?.decision === "manual") {
      reviews.set(reviewKey(sample.id, selectedFormat), {
        ...activeReview,
        expected_bbox: adjusted,
      });
      reviews = reviews;
    }
  }

  function endDrag() {
    dragging = false;
  }

  function clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), Math.max(min, max));
  }

  function styleFor(box: BBox, className: string) {
    if (!sample) return "";
    const [width, height] = sample.image_size;
    return `left:${(box[0] / width) * 100}%;top:${(box[1] / height) * 100}%;width:${((box[2] - box[0]) / width) * 100}%;height:${((box[3] - box[1]) / height) * 100}%;${className}`;
  }

  function candidateLabel(candidate?: Candidate) {
    return candidate?.bbox
      ? "available"
      : (candidate?.status ?? "not available");
  }

  function previewImageStyle(crop: BBox) {
    if (!sample) return "";
    const cropWidth = crop[2] - crop[0];
    const cropHeight = crop[3] - crop[1];
    const sourceWidth = (sample.image_size[0] / cropWidth) * 100;
    const sourceHeight = (sample.image_size[1] / cropHeight) * 100;
    return `width:${sourceWidth}%;height:${sourceHeight}%;left:${(-crop[0] / cropWidth) * 100}%;top:${(-crop[1] / cropHeight) * 100}%;`;
  }

  function next(offset: number) {
    selectedIndex = clamp(
      selectedIndex + offset,
      0,
      Math.max(0, sampleIds.length - 1),
    );
  }

  function exportReviews() {
    const reviewed = [...reviews.values()].map((review) => {
      const item = samples
        .get(review.sample_id)
        ?.formats.find((value) => value.name === review.format);
      return {
        sample_id: review.sample_id,
        format: review.format,
        decision: review.decision,
        baseline_bbox: item?.baseline.bbox,
        enhanced_bbox: item?.enhanced.bbox,
        expected_bbox: review.expected_bbox,
        reason_codes: review.reason_codes,
        note: review.note,
      };
    });
    const blob = new Blob(
      [JSON.stringify({ schema_version: 1, reviews: reviewed }, null, 2)],
      { type: "application/json" },
    );
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "icarus-review-bundle.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }
</script>

<svelte:window
  onkeydown={(event) => {
    if (event.target instanceof HTMLTextAreaElement) return;
    if (event.key === "ArrowLeft") next(-1);
    if (event.key === "ArrowRight") next(1);
    if (event.key === "1") selectedFormat = "21:9";
    if (event.key === "2") selectedFormat = "9:16";
    if (event.key === "3") selectedFormat = "9:21";
    if (event.key.toLowerCase() === "a" && baseline)
      chooseCandidate("baseline");
    if (event.key.toLowerCase() === "e" && enhanced)
      chooseCandidate("enhanced");
    if (event.key.toLowerCase() === "r") resetCrop();
  }}
/>

<main>
  <header>
    <div>
      <p class="eyebrow">Icarus crop review</p>
      <h1>Compare candidates. Commit judgment.</h1>
    </div>
    <button onclick={exportReviews} disabled={reviews.size === 0}
      >Export {reviews.size} reviews</button
    >
  </header>

  {#if loading}
    <p class="message">Loading `editor/data`…</p>
  {:else if error}
    <p class="message error">{error}</p>
  {:else if sample && format}
    <section class="toolbar">
      <button onclick={() => next(-1)} disabled={selectedIndex === 0}
        >Previous</button
      >
      <span>{selectedIndex + 1} / {sampleIds.length}</span>
      <button
        onclick={() => next(1)}
        disabled={selectedIndex === sampleIds.length - 1}>Next</button
      >
      <span class="path">{sample.original_path}</span>
      <div class="formats">
        {#each ["21:9", "9:16", "9:21"] as name}
          <button
            class:active={selectedFormat === name}
            onclick={() => (selectedFormat = name)}>{name}</button
          >
        {/each}
      </div>
    </section>

    <section class="workspace">
      <aside class="panel controls">
        <h2>Layers</h2>
        <label
          ><input type="checkbox" bind:checked={showPersons} /> Persons</label
        >
        <label><input type="checkbox" bind:checked={showFaces} /> Faces</label>
        <p>{sample.persons.length} person(s), {sample.faces.length} face(s)</p>

        <h2>Candidate</h2>
        <button onclick={() => chooseCandidate("baseline")} disabled={!baseline}
          >Accept baseline</button
        >
        <small>{candidateLabel(format.baseline)}</small>
        <button onclick={() => chooseCandidate("enhanced")} disabled={!enhanced}
          >Accept enhanced</button
        >
        <small>{candidateLabel(format.enhanced)}</small>
        <button onclick={resetCrop} disabled={!baseline}
          >Reset manual crop</button
        >
        <button class="primary" onclick={saveManual} disabled={!manualCrop}
          >Commit manual crop</button
        >
        <button onclick={() => mark("source_bound")}>Source-bound</button>
        <button onclick={() => mark("skipped")}>Skip</button>
      </aside>

      <div class="canvas-wrap">
        <div
          class="canvas"
          bind:this={canvas}
          onpointermove={dragCrop}
          onpointerup={endDrag}
          onpointercancel={endDrag}
          role="application"
          aria-label="Crop editor canvas"
        >
          <img
            src={`/data/${sample.source}`}
            alt={`Review source ${sample.original_path}`}
            draggable="false"
          />
          {#if showPersons}
            {#each sample.persons as person}
              <div class="box person" style={styleFor(person.bbox, "")}></div>
            {/each}
          {/if}
          {#if showFaces}
            {#each sample.faces as face}
              <div class="box face" style={styleFor(face.bbox, "")}></div>
            {/each}
          {/if}
          {#if baseline}
            <div class="box baseline" style={styleFor(baseline, "")}></div>
          {/if}
          {#if enhanced}
            <div class="box enhanced" style={styleFor(enhanced, "")}></div>
          {/if}
          {#if manualCrop}
            <div
              class="box manual"
              style={styleFor(manualCrop, "")}
              role="button"
              tabindex="0"
              aria-label="Move manual crop"
              onpointerdown={beginDrag}
            ></div>
          {/if}
        </div>
      </div>

      <aside class="panel review">
        <h2>Judgment</h2>
        <p>
          Current: <strong>{activeReview?.decision ?? "unreviewed"}</strong>
        </p>
        <h3>Reasons</h3>
        {#each reasonOptions as [code, label]}
          <label
            ><input
              type="checkbox"
              checked={activeDraft?.reason_codes.includes(code) ??
                activeReview?.reason_codes.includes(code) ??
                false}
              onchange={() => toggleReason(code)}
            />
            {label}</label
          >
        {/each}
        <label class="note"
          >Note<textarea
            value={activeDraft?.note ?? activeReview?.note ?? ""}
            oninput={updateNote}></textarea></label
        >
        <h3>Keys</h3>
        <p>
          <kbd>A</kbd> baseline · <kbd>E</kbd> enhanced · <kbd>R</kbd> reset
        </p>
        <p><kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> format · arrows navigate</p>
      </aside>
    </section>

    <section class="previews">
      <article>
        <h2>Baseline</h2>
        {#if baseline}
          <div
            class="preview-image"
            style={`aspect-ratio:${baseline[2] - baseline[0]} / ${baseline[3] - baseline[1]};`}
          >
            <img
              src={`/data/${sample.source}`}
              style={previewImageStyle(baseline)}
              alt="Baseline crop preview"
            />
          </div>
        {:else}<p>Unavailable</p>{/if}
      </article>
      <article>
        <h2>Enhanced</h2>
        {#if enhanced}
          <div
            class="preview-image"
            style={`aspect-ratio:${enhanced[2] - enhanced[0]} / ${enhanced[3] - enhanced[1]};`}
          >
            <img
              src={`/data/${sample.source}`}
              style={previewImageStyle(enhanced)}
              alt="Enhanced crop preview"
            />
          </div>
        {:else}<p>Unavailable</p>{/if}
      </article>
      <article>
        <h2>Preview</h2>
        {#if manualCrop}
          <div
            class="preview-image"
            style={`aspect-ratio:${manualCrop[2] - manualCrop[0]} / ${manualCrop[3] - manualCrop[1]};`}
          >
            <img
              src={`/data/${sample.source}`}
              style={previewImageStyle(manualCrop)}
              alt="Adjusted crop preview"
            />
          </div>
        {:else}<p>Unavailable</p>{/if}
      </article>
    </section>
  {/if}
</main>
