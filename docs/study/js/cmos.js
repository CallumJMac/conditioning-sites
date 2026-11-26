/**
 * Paired-comparison (CMOS / CSMOS) listening study for voice cloning evaluation.
 * Static, GitHub Pages compatible. Results POSTed to a Google Apps Script endpoint,
 * or downloaded as JSON when no endpoint is configured (dummy testing).
 *
 * URL parameters: PROLIFIC_PID, STUDY_ID, SESSION_ID (from Prolific),
 * list=L1..L8 (optional override), debug=1 (skip audio-play enforcement).
 */

const SCALE = [
    { value: -3, label: 'A much better' },
    { value: -2, label: 'A better' },
    { value: -1, label: 'A slightly better' },
    { value: 0, label: 'About the same' },
    { value: 1, label: 'B slightly better' },
    { value: 2, label: 'B better' },
    { value: 3, label: 'B much better' },
];

const QUESTIONS = {
    naturalness: 'Which sample sounds more <b>natural</b>, like a real person speaking? Ignore background noise and recording quality.',
    similarity: 'Which sample sounds more like the <b>same speaker</b> as the Reference?',
};

function shuffle(items, random) {
    const copy = items.slice();
    for (let index = copy.length - 1; index > 0; index--) {
        const swap = Math.floor(random() * (index + 1));
        [copy[index], copy[swap]] = [copy[swap], copy[index]];
    }
    return copy;
}

class CmosStudy {
    constructor(config) {
        this.config = config;
        this.container = document.getElementById('page-container');
        const params = new URLSearchParams(window.location.search);
        this.prolific = {
            pid: params.get('PROLIFIC_PID') || '',
            studyId: params.get('STUDY_ID') || '',
            sessionId: params.get('SESSION_ID') || '',
        };
        this.debug = params.get('debug') === '1';
        this.participantId = this.prolific.pid || 'local-' + Math.random().toString(36).slice(2, 10);
        const listNames = Object.keys(this.config.lists);
        this.list = params.get('list') || listNames[this.hash(this.participantId) % listNames.length];
        this.random = Math.random;
        this.startTime = Date.now();
        this.responses = [];
        this.headphone = { answers: [], passed: null };
        this.screening = {};
        this.pages = this.buildPages();
        this.pageIndex = 0;
    }

    hash(text) {
        let value = 0;
        for (const character of text) value = (value * 31 + character.charCodeAt(0)) >>> 0;
        return value;
    }

    buildPages() {
        const pages = [{ type: 'consent' }, { type: 'screening' }, { type: 'volume' }, { type: 'headphone' }, { type: 'instructions' }];
        pages.push({ type: 'practice' });
        for (const axis of shuffle(['naturalness', 'similarity'], this.random)) {
            pages.push({ type: 'block_intro', axis });
            const trials = this.config.lists[this.list][axis].map(pairId => ({ kind: 'pair', pairId }));
            for (const trap of this.config.traps) {
                trials.push({ kind: axis === 'naturalness' ? 'trap_degraded' : 'trap_identical', trapId: trap.id });
            }
            for (const trial of shuffle(trials, this.random)) pages.push({ type: 'trial', axis, ...trial });
        }
        pages.push({ type: 'feedback' }, { type: 'submit' });
        return pages;
    }

    start() { this.render(); }

    next() {
        this.pageIndex += 1;
        window.scrollTo(0, 0);
        this.render();
    }

    render() {
        const page = this.pages[this.pageIndex];
        const trialPages = this.pages.filter(item => item.type === 'trial').length;
        const done = this.pages.slice(0, this.pageIndex).filter(item => item.type === 'trial').length;
        const progress = this.pageIndex > 0
            ? `<div class="progress-bar"><div class="fill" style="width:${(done / trialPages) * 100}%"></div></div>`
            : '';
        this.container.innerHTML = progress;
        const body = document.createElement('div');
        this.container.appendChild(body);
        this[`render_${page.type}`](body, page);
    }

    render_consent(body) {
        body.innerHTML = `
            <h1>Listening study: synthetic voices</h1>
            <div class="consent-box">
                <p>You are invited to take part in a research study about computer-generated (synthetic) speech. All voices you will rate are produced by text-to-speech systems imitating speakers from public, consented speech corpora (LibriVox audiobook readers and Common Voice volunteers).</p>
                <p>You will compare pairs of short audio clips and say which sounds more natural, or which sounds more like a reference speaker. The study takes about 12 to 15 minutes. <b>You must wear headphones</b> and be in a quiet place.</p>
                <p>Participation is voluntary and you may withdraw at any time by closing this page. We store only your Prolific ID, your answers, and timing information. Data are anonymised for analysis, kept securely, and used only for research publications.</p>
                <p>Contact: callumjmac@outlook.com. This study was reviewed by the researcher against standard ethical guidelines for low-risk research: it involves anonymous adult volunteers rating audio only, collects no personal or sensitive data, and you can withdraw at any time without penalty.</p>
            </div>
            <label class="checkbox"><input type="checkbox" id="consent-check"> I have read the information above and agree to take part.</label>
            <button class="btn" id="go" disabled>Continue</button>`;
        const check = body.querySelector('#consent-check');
        const button = body.querySelector('#go');
        check.onchange = () => { button.disabled = !check.checked; };
        button.onclick = () => this.next();
    }

    render_screening(body) {
        body.innerHTML = `
            <h1>About you</h1>
            <div class="question"><p>Is English your first language?</p>
                <label><input type="radio" name="native" value="yes"> Yes</label>
                <label><input type="radio" name="native" value="no"> No</label></div>
            <div class="question"><p>Do you have any known hearing difficulties?</p>
                <label><input type="radio" name="hearing" value="no"> No</label>
                <label><input type="radio" name="hearing" value="yes"> Yes</label></div>
            <div class="question"><p>What are you listening with?</p>
                <label><input type="radio" name="device" value="over_ear"> Over-ear headphones</label>
                <label><input type="radio" name="device" value="in_ear"> In-ear headphones / earbuds</label>
                <label><input type="radio" name="device" value="speakers"> Loudspeakers</label></div>
            <p class="error" id="error"></p>
            <button class="btn" id="go">Continue</button>`;
        body.querySelector('#go').onclick = () => {
            const value = name => (body.querySelector(`input[name=${name}]:checked`) || {}).value;
            this.screening = { native: value('native'), hearing: value('hearing'), device: value('device') };
            if (Object.values(this.screening).some(item => !item)) {
                body.querySelector('#error').textContent = 'Please answer all questions.';
                return;
            }
            if (this.screening.hearing === 'yes' || this.screening.device === 'speakers') {
                return this.screenOut(body, 'This study needs headphones and no known hearing difficulties, so we cannot continue.');
            }
            this.next();
        };
    }

    render_volume(body) {
        body.innerHTML = `
            <h1>Set your volume</h1>
            <p>Put your headphones on, with both ears covered, in a quiet place.</p>
            <p>Play this clip and adjust your computer's volume until the speech is <b>clearly audible and comfortable</b>. Keep this volume for the rest of the study.</p>
            <p>The next check is hard on purpose: it plays a <b>faint tone</b> that is only audible through headphones. Listen carefully to each sound before answering.</p>
            <audio id="clip" src="${this.config.practice.reference}" preload="auto"></audio>
            <button class="btn play" id="play">Play clip</button>
            <button class="btn" id="go" ${this.debug ? '' : 'disabled'}>My volume is set</button>`;
        const clip = body.querySelector('#clip');
        body.querySelector('#play').onclick = () => { clip.currentTime = 0; clip.play(); };
        clip.onended = () => { body.querySelector('#go').disabled = false; };
        body.querySelector('#go').onclick = () => this.next();
    }

    screenOut(body, reason) {
        const url = this.config.prolificScreenedOutUrl;
        const redirect = this.prolific.pid && url;
        body.innerHTML = `<h1>Thank you</h1><p>${reason}</p>` +
            (redirect ? '<p>Returning you to Prolific...</p>' : '<p>Please return the study on Prolific.</p>');
        this.submit(true).then(() => {
            if (redirect) setTimeout(() => { window.location.href = url; }, 2500);
        });
    }

    render_headphone(body) {
        const settings = this.config.headphone;
        const trials = shuffle(settings.trials, this.random).slice(0, settings.nTrials);
        let index = 0;
        const show = () => {
            const trial = trials[index];
            body.innerHTML = `
                <h1>Headphone check (${index + 1} of ${trials.length})</h1>
                <p>Set a comfortable volume. You will hear three noises separated by short gaps. <b>One of them contains a faint tone.</b> Which one?</p>
                <audio id="clip" src="${trial.file}" preload="auto"></audio>
                <button class="btn play" id="play">Play</button>
                <div class="options">${[1, 2, 3].map(option => `<button class="btn option" data-answer="${option}" disabled>${option}</button>`).join('')}</div>`;
            const clip = body.querySelector('#clip');
            body.querySelector('#play').onclick = () => { clip.currentTime = 0; clip.play(); };
            clip.onended = () => body.querySelectorAll('.option').forEach(button => { button.disabled = false; });
            if (this.debug) body.querySelectorAll('.option').forEach(button => { button.disabled = false; });
            body.querySelectorAll('.option').forEach(button => {
                button.onclick = () => {
                    this.headphone.answers.push({ file: trial.file, answer: Number(button.dataset.answer), correct: Number(button.dataset.answer) === trial.answer });
                    index += 1;
                    if (index < trials.length) return show();
                    const correct = this.headphone.answers.filter(item => item.correct).length;
                    this.headphone.passed = correct >= settings.passThreshold;
                    if (this.headphone.passed) return this.next();
                    this.screenOut(body, 'Unfortunately the headphone check was not passed, so we cannot continue. This usually means headphones were not used.');
                };
            });
        };
        show();
    }

    render_instructions(body) {
        body.innerHTML = `
            <h1>Instructions</h1>
            <p>On each page you will hear two clips, <b>A</b> and <b>B</b>, saying the same sentence. Listen to both fully (you may replay them) and then choose an answer on the scale.</p>
            <p>There are two parts:</p>
            <ul><li><b>Naturalness</b>: which clip sounds more like a real person speaking naturally? Ignore background noise and recording quality.</li>
            <li><b>Speaker match</b>: first listen to a <b>Reference</b> clip of a speaker; then judge which of A or B sounds more like that same person. Ignore what is being said.</li></ul>
            <p>Differences can be small. If you genuinely cannot hear a difference, choose "About the same", but please give your best judgement.</p>
            <button class="btn" id="go">Start practice</button>`;
        body.querySelector('#go').onclick = () => this.next();
    }

    render_practice(body) {
        const practice = this.config.practice;
        this.renderComparison(body, {
            title: 'Practice',
            question: QUESTIONS.similarity,
            reference: practice.reference,
            a: practice.a,
            b: practice.b,
            onAnswer: () => this.next(),
        });
    }

    render_block_intro(body, page) {
        const text = page.axis === 'naturalness'
            ? 'In this part, judge <b>naturalness</b>: which clip sounds more like a real person speaking? Ignore background noise and recording quality.'
            : 'In this part, judge <b>speaker match</b>: listen to the Reference, then decide which clip sounds more like the same person.';
        body.innerHTML = `<h1>${page.axis === 'naturalness' ? 'Part: Naturalness' : 'Part: Speaker match'}</h1><p>${text}</p><button class="btn" id="go">Begin</button>`;
        body.querySelector('#go').onclick = () => this.next();
    }

    render_trial(body, page) {
        let a, b, reference = null, meta;
        if (page.kind === 'pair') {
            const pair = this.config.pairs[page.pairId];
            const iclFirst = this.random() < 0.5;
            a = iclFirst ? pair.icl : pair.xvec;
            b = iclFirst ? pair.xvec : pair.icl;
            meta = { pairId: page.pairId, a: iclFirst ? 'icl' : 'xvec', b: iclFirst ? 'xvec' : 'icl' };
        } else {
            const trap = this.config.traps.find(item => item.id === page.trapId);
            if (page.kind === 'trap_degraded') {
                const naturalFirst = this.random() < 0.5;
                a = naturalFirst ? trap.natural : trap.degraded;
                b = naturalFirst ? trap.degraded : trap.natural;
                meta = { trapId: trap.id, a: naturalFirst ? 'natural' : 'degraded', b: naturalFirst ? 'degraded' : 'natural' };
            } else {
                a = trap.same;
                b = trap.same;
                meta = { trapId: trap.id, a: 'same', b: 'same' };
            }
        }
        if (page.axis === 'similarity') {
            reference = page.kind === 'pair' ? this.config.pairs[page.pairId].reference : this.config.traps.find(item => item.id === page.trapId).reference;
        }
        const trialPages = this.pages.filter(item => item.type === 'trial');
        const number = trialPages.indexOf(page) + 1;
        const shownAt = Date.now();
        this.renderComparison(body, {
            title: `Comparison ${number} of ${trialPages.length}`,
            question: QUESTIONS[page.axis],
            reference, a, b,
            onAnswer: (score, plays) => {
                this.responses.push({ axis: page.axis, kind: page.kind, ...meta, score, plays, responseMs: Date.now() - shownAt });
                this.next();
            },
        });
    }

    renderComparison(body, { title, question, reference, a, b, onAnswer }) {
        const clips = [];
        if (reference) clips.push(['Reference', reference]);
        clips.push(['A', a], ['B', b]);
        body.innerHTML = `
            ${title ? `<h1>${title}</h1>` : ''}
            <p class="question-text">${question}</p>
            <div class="players">${clips.map(([label, src]) => `
                <div class="player"><span class="label">${label}</span>
                <audio src="${src}" preload="auto" data-label="${label}"></audio>
                <button class="btn play" data-label="${label}">Play ${label}</button></div>`).join('')}</div>
            <div class="scale">${SCALE.map(option => `<button class="btn option" data-value="${option.value}" disabled>${option.label}</button>`).join('')}</div>
            <p class="hint" id="hint">Play every clip to the end to enable the answer buttons.</p>`;
        const plays = {};
        const finished = new Set();
        const audios = [...body.querySelectorAll('audio')];
        const enable = () => {
            if (this.debug || clips.every(([label]) => finished.has(label))) {
                body.querySelectorAll('.option').forEach(button => { button.disabled = false; });
                body.querySelector('#hint').textContent = '';
            }
        };
        body.querySelectorAll('.play').forEach(button => {
            button.onclick = () => {
                audios.forEach(audio => audio.pause());
                const audio = audios.find(item => item.dataset.label === button.dataset.label);
                audio.currentTime = 0;
                audio.play();
                plays[button.dataset.label] = (plays[button.dataset.label] || 0) + 1;
            };
        });
        audios.forEach(audio => { audio.onended = () => { finished.add(audio.dataset.label); enable(); }; });
        enable();
        body.querySelectorAll('.option').forEach(button => {
            button.onclick = () => { audios.forEach(audio => audio.pause()); onAnswer(Number(button.dataset.value), plays); };
        });
    }

    render_feedback(body) {
        body.innerHTML = `
            <h1>Almost done</h1>
            <p>Any comments (optional)? For example, problems with audio, or what you listened for.</p>
            <textarea id="comments" rows="4"></textarea>
            <button class="btn" id="go">Submit</button>`;
        body.querySelector('#go').onclick = () => {
            this.comments = body.querySelector('#comments').value;
            this.next();
        };
    }

    render_submit(body) {
        body.innerHTML = `<h1>Submitting...</h1><p>Please wait and do not close this page.</p>`;
        this.submit(false).then(() => {
            const url = this.config.prolificCompletionUrl;
            body.innerHTML = `<h1>Thank you!</h1><p>Your responses have been recorded.</p>` +
                (this.prolific.pid ? `<p>Returning you to Prolific...</p>` : `<p>(Local test: results downloaded as JSON.)</p>`);
            if (this.prolific.pid) setTimeout(() => { window.location.href = url; }, 1500);
        });
    }

    async submit(screenedOut) {
        const payload = {
            testId: this.config.testId,
            participantId: this.participantId,
            prolific: this.prolific,
            list: this.list,
            screenedOut,
            screening: this.screening,
            headphone: this.headphone,
            responses: this.responses,
            comments: this.comments || '',
            durationMs: Date.now() - this.startTime,
            userAgent: navigator.userAgent,
            submittedAt: new Date().toISOString(),
        };
        const download = () => {
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `results_${this.participantId}.json`;
            link.click();
        };
        if (!this.config.resultsEndpoint) {
            download();
            return;
        }
        // text/plain avoids a CORS preflight, which Apps Script does not answer.
        // If the upload fails, fall back to a download so no results are lost.
        await fetch(this.config.resultsEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'text/plain;charset=utf-8' },
            body: JSON.stringify(payload),
        }).catch(download);
    }
}
