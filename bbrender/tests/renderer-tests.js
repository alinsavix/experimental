export function runRendererTests(BBRender, doc = document) {
  const results = [];

  function assert(name, condition) {
    results.push({ name, passed: !!condition });
  }

  function section(name, fn) {
    try {
      fn();
    } catch (error) {
      results.push({
        name,
        passed: false,
        error: error && error.message ? error.message : String(error)
      });
    }
  }

  section('parser basics', () => {
    const parsed = BBRender.parse('[b]hello[/b]');
    assert('parser creates a b tag', parsed.ast.children[0].name === 'b');

    const attrs = BBRender.parseTagToken('gradient stops="red 0%,gold 50%" speed=2');
    assert('quoted attributes parse', attrs.attrs.stops === 'red 0%,gold 50%');
    assert('numeric-looking attributes stay strings', attrs.attrs.speed === '2');

    const noparse = BBRender.parse('[noparse][b]x[/b][/noparse]');
    assert('noparse keeps literal tags', noparse.ast.children[0].value === '[b]x[/b]');

    const reset = BBRender.parse('[b]x[reset]y');
    assert('reset returns to root', reset.ast.children.length === 2);
  });

  section('basic rendering', () => {
    const holder = doc.createElement('div');
    BBRender.renderToElement(holder, '[b]hello[/b][newline]world');
    assert('renderer creates tag span', holder.querySelector('[data-bb-tag="b"]') !== null);
    assert('renderer creates newline br', holder.querySelector('br') !== null);
    assert('renderer preserves text', holder.textContent === 'helloworld');
  });

  section('inline styles', () => {
    const styled = doc.createElement('div');
    BBRender.renderToElement(styled, '[b]x[/b][color=dodger-blue]y[/color][hr]');
    assert('bold style applies', styled.querySelector('[data-bb-tag="b"]').style.fontWeight === '700');
    assert('hyphenated color name normalizes', styled.querySelector('[data-bb-tag="color"]').style.color !== '');
    assert('hr renders as rule element', styled.querySelector('hr') !== null);

    const stroked = doc.createElement('div');
    BBRender.renderToElement(stroked, '[stroke color=black width=3]x[/stroke]');
    assert('stroke tag applies text stroke', stroked.querySelector('[data-bb-tag="stroke"]').style.webkitTextStroke === '3px black');

    const spaced = doc.createElement('div');
    BBRender.renderToElement(spaced, '[b][i]Styled[/all] plain [color=dodger-blue]Blue[/color] [opacity=0.45]Faded[/opacity]');
    assert('renderer preserves spaces between styled spans', spaced.textContent.includes('Styled plain Blue Faded'));
  });

  section('animations', () => {
    const animated = doc.createElement('div');
    BBRender.renderToElement(animated, '[wave][blink]ok[/blink][/wave]');
    assert('animation creates character wrappers', animated.querySelectorAll('.bb-char').length >= 2);
    assert('nested animation wrappers are present', animated.querySelector('[data-bb-anim="wave"] [data-bb-anim="blink"]') !== null);

    const blinked = doc.createElement('div');
    BBRender.renderToElement(blinked, '[blink]o k[/blink]');
    const blinkWrappers = Array.from(blinked.querySelectorAll('[data-bb-anim="blink"]'));
    assert('blink wrappers exist', blinkWrappers.length === 3);
    assert('blink is synchronized', blinkWrappers.every((el) => el.style.getPropertyValue('--bb-delay') === '0s'));
    assert('animated spaces are preserved', blinked.querySelector('.bb-space') !== null);

    const flipped = doc.createElement('div');
    BBRender.renderToElement(flipped, '[flip axis=x]x[/flip][flip axis=y]y[/flip]');
    assert('flip x animation is assigned', flipped.querySelector('[data-bb-anim="flip"]').style.animation.includes('bb-flip-x'));
    assert('flip y animation is assigned', flipped.querySelectorAll('[data-bb-anim="flip"]')[1].style.animation.includes('bb-flip-y'));

    const spun = doc.createElement('div');
    BBRender.renderToElement(spun, '[rotate speed=45]go[/rotate]');
    assert('rotate creates animation wrappers', spun.querySelectorAll('[data-bb-anim="rotate"]').length === 2);
    assert('rotate animation is assigned', spun.querySelector('[data-bb-anim="rotate"]').style.animation.includes('bb-rotate'));

    const metallic = doc.createElement('div');
    BBRender.renderToElement(metallic, '[metallic speed=2][color=gold]Au[/color][/metallic]');
    assert('metallic creates animation wrappers', metallic.querySelectorAll('[data-bb-anim="metallic"]').length === 2);
    assert('metallic sheen animation is assigned', metallic.querySelector('[data-bb-anim="metallic"]').style.animation.includes('bb-metallic'));
    assert('metallic clips gradient to text', metallic.querySelector('[data-bb-anim="metallic"]').style.webkitBackgroundClip === 'text');
  });

  section('character effects', () => {
    const typed = doc.createElement('div');
    BBRender.renderToElement(typed, '[typewriter speed=8 cursor=1]ab[/typewriter]');
    assert('typewriter creates effect wrappers', typed.querySelectorAll('[data-bb-effect="typewriter"]').length === 2);
    assert('typewriter assigns reveal animation', typed.querySelector('[data-bb-effect="typewriter"]').style.animation.includes('bb-typewriter'));
    assert('typewriter cursor marker is present', typed.querySelector('[data-bb-tag="typewriter"][data-bb-attr-cursor="1"]') !== null);
    assert('typewriter cursor starts on first character', typed.querySelectorAll('[data-bb-effect="typewriter"]')[0].classList.contains('bb-typewriter-cursor'));

    const hacked = doc.createElement('div');
    BBRender.renderToElement(hacked, '[hacker speed=120 loop=0 seed=test glyphs=Z]A[/hacker]');
    assert('hacker creates effect wrapper', hacked.querySelector('[data-bb-effect="hacker"]') !== null);
    assert('hacker mutates text deterministically before reveal', hacked.textContent === 'Z');

    const faded = doc.createElement('div');
    BBRender.renderToElement(faded, '[fade start=1 length=2]abc[/fade]');
    const fadedChars = faded.querySelectorAll('[data-bb-effect="fade"]');
    assert('fade creates per-character wrappers', fadedChars.length === 3);
    assert('fade opacity ramp starts after configured index', fadedChars[0].style.opacity === '1' && fadedChars[2].style.opacity === '0.5');

    const burning = doc.createElement('div');
    BBRender.renderToElement(burning, '[fire intensity=0.5]hot[/fire]');
    assert('fire creates effect wrappers', burning.querySelectorAll('[data-bb-effect="fire"]').length === 3);
    assert('fire animation is assigned', burning.querySelector('[data-bb-effect="fire"]').style.animation.includes('bb-fire'));

    const zapped = doc.createElement('div');
    BBRender.renderToElement(zapped, '[electric freq=10 intensity=5]zap[/electric]');
    assert('electric creates effect wrappers', zapped.querySelectorAll('[data-bb-effect="electric"]').length === 3);
    assert('electric animation is assigned', zapped.querySelector('[data-bb-effect="electric"]').style.animation.includes('bb-electric'));
  });

  section('motion and layout', () => {
    const moved = doc.createElement('div');
    BBRender.renderToElement(moved, '[slide dir=right speed=80]x[/slide][zoom from=0 to=1 speed=2]y[/zoom]');
    assert('slide assigns entrance animation', moved.querySelector('[data-bb-tag="slide"]').style.animation.includes('bb-slide-in'));
    assert('zoom assigns scale animation', moved.querySelector('[data-bb-tag="zoom"]').style.animation.includes('bb-zoom'));

    const wrapped = doc.createElement('div');
    BBRender.renderToElement(wrapped, '[wrap=char]longword[/wrap]', {
      layout: { width: 240, height: 120, padding: 10, lineHeight: 1.25 }
    });
    assert('wrap tag applies character wrapping', wrapped.querySelector('[data-bb-tag="wrap"]').style.overflowWrap === 'anywhere');
    assert('source width layout applies', wrapped.style.width === '240px');
    assert('source height layout applies', wrapped.style.height === '120px');
    assert('manual padding layout applies', wrapped.style.padding === '10px');
    assert('line height layout applies', wrapped.style.lineHeight === '1.25');

    const autoPadded = doc.createElement('div');
    BBRender.renderToElement(autoPadded, '[wave amp=42]pad[/wave]', {
      layout: { autoPadding: true }
    });
    assert('auto padding estimates animated bounds', parseInt(autoPadded.style.padding, 10) >= 50);
  });

  section('root transitions', () => {
    const transitioned = doc.createElement('div');
    BBRender.renderToElement(transitioned, 'Lifecycle text', {
      duration: 5000,
      transition: { in: 'zoom', out: 'fade', inTime: 400, outTime: 600 }
    });
    assert('root transition marker is assigned', transitioned.dataset.bbTransition === 'root');
    assert('root transition includes entrance animation', transitioned.style.animation.includes('bb-enter-zoom') && transitioned.style.animation.includes('400ms'));
    assert('root transition includes exit animation', transitioned.style.animation.includes('bb-exit-fade') && transitioned.style.animation.includes('600ms'));
    assert('root transition computes exit delay from duration', transitioned.style.animation.includes('4400ms'));
    assert('root transition sets zoom scale default', transitioned.style.getPropertyValue('--bb-enter-zoom-scale') === '0.18');

    BBRender.renderToElement(transitioned, 'No transition');
    assert('root transition clears on next render', transitioned.dataset.bbTransition === undefined && transitioned.style.animation === '' && transitioned.style.getPropertyValue('--bb-enter-zoom-scale') === '');
  });

  section('dropcap fonts and images', () => {
    const dropped = doc.createElement('div');
    BBRender.renderToElement(dropped, '[dropcap lines=4]Alpha beta[/dropcap]');
    assert('dropcap wraps first grapheme', dropped.querySelector('.bb-dropcap-first').textContent === 'A');
    assert('dropcap line count is assigned', dropped.querySelector('[data-bb-tag="dropcap"]').style.getPropertyValue('--bb-dropcap-lines') === '4');

    const fonted = doc.createElement('div');
    BBRender.renderToElement(fonted, '[gfont=Lobster]font[/gfont]');
    assert('google font span applies family', fonted.querySelector('[data-bb-tag="gfont"]').style.fontFamily.includes('Lobster'));
    assert('google font stylesheet link is injected', doc.getElementById('bb-gfont-lobster') !== null);

    const imaged = doc.createElement('div');
    BBRender.renderToElement(imaged, '[img src="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27/%3E" width=32 height=24]');
    const img = imaged.querySelector('img.bb-img');
    assert('image tag creates img element', img !== null);
    assert('image dimensions apply', img.style.width === '32px' && img.style.height === '24px');

    const randomed = doc.createElement('div');
    BBRender.renderToElement(randomed, '[random words="Hello,Hi,Hey" speed=2]');
    assert('random tag creates dynamic span', randomed.querySelector('[data-bb-tag="random"]') !== null);
    assert('random tag chooses one configured word', ['Hello', 'Hi', 'Hey'].includes(randomed.textContent));
  });

  section('diagnostics and payloads', () => {
    const unknownDiagnostics = [];
    BBRender.renderToElement(doc.createElement('div'), '[wat]x[/wat]', {
      onDiagnostics: (items) => unknownDiagnostics.push(...items)
    });
    assert('unknown tags report diagnostics', unknownDiagnostics.length >= 1);

    assert('bbcode.render payload is handled', BBRender.shouldHandlePayload({ type: 'bbcode.render', bbcode: 'x' }));
    assert('custom type comparison is case-insensitive', BBRender.shouldHandlePayload({ type: 'BBCode.Render', bbcode: 'x' }));
    assert('unrelated custom payload is ignored', !BBRender.shouldHandlePayload({ type: 'other-client', bbcode: 'x' }));
    assert('untyped payload is ignored', !BBRender.shouldHandlePayload({ bbcode: 'x' }));
  });

  const failures = results.filter((result) => !result.passed);
  return {
    passed: results.length - failures.length,
    failed: failures.length,
    failures,
    results
  };
}

export function renderRendererTestResults(summary, doc = document) {
  const list = doc.createElement('div');
  list.className = 'test-results';

  const heading = doc.createElement('p');
  heading.className = summary.failed ? 'test-fail' : 'test-pass';
  heading.textContent = `${summary.passed} passed, ${summary.failed} failed`;
  list.appendChild(heading);

  summary.results.forEach((result) => {
    const row = doc.createElement('p');
    row.className = result.passed ? 'test-pass' : 'test-fail';
    row.textContent = (result.passed ? 'PASS ' : 'FAIL ') + result.name + (result.error ? ': ' + result.error : '');
    list.appendChild(row);
  });

  return list;
}
