<script lang="ts">
  import { onMount } from "svelte";

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
  type Draft = Pick<Review, "reason_codes" | "note">;
  type PersistedSession = {
    sample_id?: string;
    format?: string;
    reviews: [string, Review][];
    drafts: [string, Draft][];
    manual_crops: [string, BBox][];
  };

  const SESSION_STORAGE_KEY = "icarus-editor.session.v1";
  const reasonOptions = [
    ["face_cut", "Face cut", "The crop cuts away part of the detected face."],
    [
      "face_edge_tight",
      "Face edge-tight",
      "The full face is visible but uncomfortably close to a crop edge.",
    ],
    [
      "subject_too_high",
      "Subject too high",
      "Move the crop down to place the subject lower.",
    ],
    [
      "subject_too_low",
      "Subject too low",
      "Move the crop up to place the subject higher.",
    ],
    [
      "subject_too_left",
      "Subject too left",
      "Move the crop right to place the subject closer to center.",
    ],
    [
      "subject_too_right",
      "Subject too right",
      "Move the crop left to place the subject closer to center.",
    ],
    [
      "wrong_subject",
      "Wrong subject",
      "The candidate frames a different detected person than the intended subject.",
    ],
    [
      "group_framing",
      "Group framing",
      "Judge this as a deliberate group composition, not a single-person crop.",
    ],
    [
      "detector_error",
      "Detector error",
      "Person or face detection box is wrong, so crop behavior cannot be fairly judged.",
    ],
  ] as const;

  let sampleIds: string[] = [];
  let samples = new Map<string, Sample>();
  let selectedIndex = 0;
  let selectedFormat = "21:9";
  let manualCrop: BBox | undefined;
  let manualCrops = new Map<string, BBox>();
  let reviews = new Map<string, Review>();
  let drafts = new Map<string, Draft>();
  let restoredSession: PersistedSession | undefined;
  let showPersons = true;
  let showFaces = true;
  let loading = true;
  let error = "";
  let dragging = false;
  let dragOffset: [number, number] = [0, 0];
  let canvas: HTMLDivElement;
  let hoveredPreview: { label: string; crop: BBox } | undefined;
  let showPreviewPopup = false;

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
  $: if (sample && format) restoreManualCrop();

  onMount(() => {
    restoredSession = loadSession();
    if (!loading) restoreSessionSelection();
  });
  loadManifest();

  async function loadManifest() {
    try {
      const manifest = await fetch("/data/manifest.json").then(requireOk);
      sampleIds = manifest.samples;
      await Promise.all(sampleIds.map(loadSample));
      restoreSessionSelection();
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

  function loadSession(): PersistedSession | undefined {
    try {
      const raw = localStorage.getItem(SESSION_STORAGE_KEY);
      return raw ? (JSON.parse(raw) as PersistedSession) : undefined;
    } catch {
      return undefined;
    }
  }

  function restoreSessionSelection() {
    if (!restoredSession) return;
    reviews = new Map(restoredSession.reviews ?? []);
    drafts = new Map(restoredSession.drafts ?? []);
    manualCrops = new Map(restoredSession.manual_crops ?? []);
    if (
      restoredSession.format &&
      ["21:9", "9:16", "9:21"].includes(restoredSession.format)
    ) {
      selectedFormat = restoredSession.format;
    }
    if (restoredSession.sample_id) {
      const index = sampleIds.indexOf(restoredSession.sample_id);
      if (index >= 0) selectedIndex = index;
    }
  }

  function persistSession() {
    try {
      localStorage.setItem(
        SESSION_STORAGE_KEY,
        JSON.stringify({
          sample_id: selectedId,
          format: selectedFormat,
          reviews: [...reviews.entries()],
          drafts: [...drafts.entries()],
          manual_crops: [...manualCrops.entries()],
        } satisfies PersistedSession),
      );
    } catch {
      // Browser storage can be unavailable or full. Review export still works.
    }
  }

  function restoreManualCrop() {
    if (!sample) return;
    const saved = manualCrops.get(reviewKey(sample.id, selectedFormat));
    manualCrop = saved
      ? ([...saved] as BBox)
      : baseline
        ? ([...baseline] as BBox)
        : undefined;
  }

  function resetCrop() {
    if (!sample) return;
    const key = reviewKey(sample.id, selectedFormat);
    manualCrops.delete(key);
    manualCrops = manualCrops;
    manualCrop = baseline ? ([...baseline] as BBox) : undefined;
    persistSession();
  }

  function chooseCandidate(kind: "baseline" | "enhanced") {
    const candidate = kind === "baseline" ? baseline : enhanced;
    if (!candidate || !sample) return;
    manualCrop = [...candidate] as BBox;
    manualCrops.set(reviewKey(sample.id, selectedFormat), manualCrop);
    manualCrops = manualCrops;
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
    persistSession();
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
    persistSession();
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
    persistSession();
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
    persistSession();
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

  function previewClass(crop: BBox) {
    return crop[2] - crop[0] >= crop[3] - crop[1] ? "landscape" : "portrait";
  }

  function previewContainerStyle(crop: BBox) {
    const aspect = (crop[2] - crop[0]) / (crop[3] - crop[1]);
    return `aspect-ratio:${aspect};--aspect:${aspect};`;
  }

  function showPreviewImmediately(label: string, crop: BBox) {
    hoveredPreview = { label, crop };
    showPreviewPopup = true;
  }

  function hidePreviewImmediately() {
    showPreviewPopup = false;
    hoveredPreview = undefined;
  }

  function showPreviewFromKey(event: KeyboardEvent, label: string, crop: BBox) {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    showPreviewImmediately(label, crop);
  }

  function hidePreviewFromKey(event: KeyboardEvent) {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    hidePreviewImmediately();
  }

  function dismissPreviewOnPointerMove() {
    if (showPreviewPopup) hidePreviewImmediately();
  }

  function next(offset: number) {
    selectedIndex = clamp(
      selectedIndex + offset,
      0,
      Math.max(0, sampleIds.length - 1),
    );
    persistSession();
  }

  function selectFormat(format: string) {
    selectedFormat = format;
    persistSession();
  }

  function exportReviews() {
    const reviewed = [...reviews.values()]
      .filter((review) => review.decision !== "skipped")
      .map((review) => {
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
  onpointermove={dismissPreviewOnPointerMove}
  onkeydown={(event) => {
    if (event.target instanceof HTMLTextAreaElement) return;
    if (event.key === "ArrowLeft") next(-1);
    if (event.key === "ArrowRight") next(1);
    if (event.key === "1") selectFormat("21:9");
    if (event.key === "2") selectFormat("9:16");
    if (event.key === "3") selectFormat("9:21");
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
            onclick={() => selectFormat(name)}>{name}</button
          >
        {/each}
      </div>
    </section>

    <section class="previews" aria-label="Crop candidates">
      {#if baseline}
        <div
          class="preview-card"
          onclick={() => showPreviewImmediately("Baseline", baseline)}
          onkeydown={(event) => showPreviewFromKey(event, "Baseline", baseline)}
          role="button"
          tabindex="0"
          aria-label="Show Baseline crop preview"
        >
          <h2>Baseline</h2>
          <div
            class="preview-image"
            class:portrait={previewClass(baseline) === "portrait"}
            style={previewContainerStyle(baseline)}
          >
            <img
              src={`/data/${sample.source}`}
              style={previewImageStyle(baseline)}
              alt="Baseline crop preview"
            />
          </div>
        </div>
      {/if}
      {#if enhanced}
        <div
          class="preview-card"
          onclick={() => showPreviewImmediately("Enhanced", enhanced)}
          onkeydown={(event) => showPreviewFromKey(event, "Enhanced", enhanced)}
          role="button"
          tabindex="0"
          aria-label="Show Enhanced crop preview"
        >
          <h2>Enhanced</h2>
          <div
            class="preview-image"
            class:portrait={previewClass(enhanced) === "portrait"}
            style={previewContainerStyle(enhanced)}
          >
            <img
              src={`/data/${sample.source}`}
              style={previewImageStyle(enhanced)}
              alt="Enhanced crop preview"
            />
          </div>
        </div>
      {/if}
      {#if manualCrop}
        <div
          class="preview-card"
          onclick={() => showPreviewImmediately("Adjusted", manualCrop!)}
          onkeydown={(event) =>
            showPreviewFromKey(event, "Adjusted", manualCrop!)}
          role="button"
          tabindex="0"
          aria-label="Show Adjusted crop preview"
        >
          <h2>Adjusted</h2>
          <div
            class="preview-image"
            class:portrait={previewClass(manualCrop) === "portrait"}
            style={previewContainerStyle(manualCrop)}
          >
            <img
              src={`/data/${sample.source}`}
              style={previewImageStyle(manualCrop)}
              alt="Adjusted crop preview"
            />
          </div>
        </div>
      {/if}
    </section>

    <section class="workspace">
      <aside class="panel controls">
        <h2>Layers</h2>
        <label
          ><input type="checkbox" bind:checked={showPersons} /> Persons</label
        >
        <label><input type="checkbox" bind:checked={showFaces} /> Faces</label>
        <p>{sample.persons.length} person(s), {sample.faces.length} face(s)</p>

        <h2>Decision</h2>
        <button onclick={() => chooseCandidate("baseline")} disabled={!baseline}
          >Accept baseline</button
        >
        <small
          >Use current baseline crop. {candidateLabel(format.baseline)}</small
        >
        <button onclick={() => chooseCandidate("enhanced")} disabled={!enhanced}
          >Accept enhanced</button
        >
        <small
          >Use current enhanced crop. {candidateLabel(format.enhanced)}</small
        >
        <button onclick={resetCrop} disabled={!baseline}
          >Reset manual crop</button
        >
        <small>Discard local drag changes. Restore baseline rectangle.</small>
        <button onclick={saveManual} disabled={!manualCrop}
          >Commit manual crop</button
        >
        <small>Save current red rectangle as intended crop.</small>
        <button onclick={() => mark("source_bound")}>Mark source-bound</button>
        <small
          >Current source lacks pixels for a better crop. Keep this as a
          reviewed limitation.</small
        >
        <button onclick={() => mark("skipped")}>Skip without fixture</button>
        <small
          >Defer this format. It is remembered, but excluded from exported
          reviews.</small
        >
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
        {#each reasonOptions as [code, label, description]}
          <label title={description}
            ><input
              type="checkbox"
              checked={activeDraft?.reason_codes.includes(code) ??
                activeReview?.reason_codes.includes(code) ??
                false}
              onchange={() => toggleReason(code)}
            />
            {label}<span class="reason-help">{description}</span></label
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
  {/if}
</main>

{#if showPreviewPopup && hoveredPreview && sample}
  <div class="preview-popup" role="presentation">
    <div
      class={`preview-popup-content ${previewClass(hoveredPreview.crop)}`}
      style={previewContainerStyle(hoveredPreview.crop)}
      onclick={hidePreviewImmediately}
      onkeydown={hidePreviewFromKey}
      role="button"
      tabindex="0"
      aria-label="Hide enlarged crop preview"
    >
      <p>{hoveredPreview.label}</p>
      <div class="preview-image">
        <img
          src={`/data/${sample.source}`}
          style={previewImageStyle(hoveredPreview.crop)}
          alt={`${hoveredPreview.label} crop preview`}
        />
      </div>
    </div>
  </div>
{/if}
