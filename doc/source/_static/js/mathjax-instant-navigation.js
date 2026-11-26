/*
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// With the theme's "navigation.instant" feature, internal links swap in the
// next page's content without a full page load, so MathJax never typesets the
// math on it. The theme emits every new page on `document$`, which is how
// mkdocs-material (the theme's upstream) documents integrating MathJax:
// https://squidfunk.github.io/mkdocs-material/reference/math/#mathjax
(() => {
  // The theme tells which scripts are already loaded by their absolute URL,
  // but only makes `src` absolute for the scripts that come before its own
  // bundle. Without this, navigating to a page in another directory loads
  // MathJax (and this file) a second time, and the second MathJax fails.
  for (const script of document.querySelectorAll('script[src]')) {
    script.src = script.src;
  }

  let initialPage = true;
  let queued = false;

  document$.subscribe(() => {
    // MathJax typesets the page it is loaded on by itself. Typesetting the
    // same math again would also typeset MathJax's own hidden MathML copy.
    if (initialPage) {
      initialPage = false;
      return;
    }
    // MathJax is only included on pages with math. If this navigation added
    // it, it typesets the new page by itself once it has loaded.
    if (!window.MathJax || !MathJax.typesetPromise || queued) return;

    // Wait for any typesetting in progress, as MathJax recommends.
    queued = true;
    MathJax.startup.promise = MathJax.startup.promise
      .then(() => {
        queued = false;
        // The navigation removed MathJax's stylesheet from <head>; clearing the
        // cache makes MathJax build a complete one for the new page.
        MathJax.startup.output.clearCache();
        MathJax.typesetClear();
        MathJax.texReset();
        return MathJax.typesetPromise();
      })
      .catch((err) => console.error('MathJax typesetting failed:', err));
  });
})();
