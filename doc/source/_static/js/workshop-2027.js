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

(() => {
  // JuMP maintains the schedule. Use its published branch first; the PR branch
  // makes the program visible here while the shared event page is under review.
  const scheduleSources = [
    'https://raw.githubusercontent.com/jump-dev/jump-dev.github.io/master/' +
      '_includes/jump-dev-2027-schedule.html',
    'https://raw.githubusercontent.com/jump-dev/jump-dev.github.io/od/jump-dev-2027/' +
      '_includes/jump-dev-2027-schedule.html',
  ];

  async function loadSchedule() {
    for (const source of scheduleSources) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000);
      try {
        const response = await fetch(source, { signal: controller.signal });
        if (!response.ok) continue;
        const document = new DOMParser().parseFromString(await response.text(), 'text/html');
        const sourceTable = document.querySelector('table');
        if (sourceTable && sourceTable.rows.length > 1) return sourceTable;
      } catch (_) {
        // Try the next authoritative source when this one is unavailable.
      } finally {
        clearTimeout(timeout);
      }
    }
    return null;
  }

  function renderSchedule(sourceTable, container) {
    const table = document.createElement('table');
    table.setAttribute('aria-label', 'Shared workshop schedule, July 22–23, 2027');
    const head = document.createElement('thead');
    const body = document.createElement('tbody');

    for (const [rowIndex, sourceRow] of Array.from(sourceTable.rows).entries()) {
      const row = document.createElement('tr');
      for (const [cellIndex, sourceCell] of Array.from(sourceRow.cells).entries()) {
        const isHeading = rowIndex === 0 || cellIndex === 0;
        const cell = document.createElement(isHeading ? 'th' : 'td');
        cell.colSpan = sourceCell.colSpan;
        cell.rowSpan = sourceCell.rowSpan;
        if (isHeading) cell.scope = rowIndex === 0 ? 'col' : 'row';
        if (sourceCell.classList.contains('talk-break')) cell.classList.add('break-session');
        if (/CVXPY/i.test(sourceCell.textContent)) cell.classList.add('cvxpy-session');

        const title = sourceCell.querySelector('.talk-title');
        const speaker = sourceCell.querySelector('.talk-speaker');
        if (rowIndex === 0 && cellIndex === 0) {
          cell.textContent = 'Time';
        } else if (title) {
          const titleText = document.createElement('strong');
          titleText.textContent = title.textContent.trim();
          cell.append(titleText);
        } else if (!speaker) {
          cell.textContent = sourceCell.textContent.trim();
        }
        if (speaker) {
          const speakerText = document.createElement('span');
          speakerText.className = 'talk-speaker';
          speakerText.textContent = speaker.textContent.trim();
          cell.append(speakerText);
        }
        row.append(cell);
      }
      if (rowIndex === 0) head.append(row);
      else body.append(row);
    }
    table.append(head, body);
    container.replaceChildren(table);
  }

  async function initializeSchedule() {
    const container = document.getElementById('workshop-2027-schedule');
    if (!container) return;
    const sourceTable = await loadSchedule();
    if (!container.isConnected) return;
    if (sourceTable) {
      renderSchedule(sourceTable, container);
    } else {
      const message = document.createElement('p');
      const link = document.createElement('a');
      link.href = 'https://jump.dev/meetings/jumpdev2027/';
      link.textContent = 'View the schedule on the shared event website';
      message.append('The schedule could not be loaded. ', link, '.');
      container.replaceChildren(message);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSchedule, { once: true });
  } else {
    initializeSchedule();
  }
})();
