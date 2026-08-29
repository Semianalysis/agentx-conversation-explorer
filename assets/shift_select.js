/* Shift-click range selection for the conversation lists' checkbox column.
 *
 * Dash DataTable has no native shift-click, so this delegates on document:
 * remember the last checkbox row the user clicked per table (in DISPLAYED
 * order, i.e. after sort/filter); on a shift-click, replay real .click()s on
 * every checkbox between anchor and target whose state differs from the
 * clicked box's new state. Real clicks go through React, so selected_rows
 * updates exactly as if the user clicked each row.
 */
(function () {
  "use strict";
  var TABLES = "#at-explorer-conv-table, #at-deep-conv-table";
  var anchors = {}; // table id -> last human-clicked index into visible checkboxes
  var synthetic = false;

  function visibleCheckboxes(container) {
    return Array.prototype.filter.call(
      container.querySelectorAll("input[type=checkbox]"),
      function (b) { return b.closest("tr") && !b.disabled; }
    );
  }

  document.addEventListener("click", function (e) {
    if (synthetic) return;
    var cb = e.target;
    if (!(cb instanceof HTMLInputElement) || cb.type !== "checkbox") return;
    var container = cb.closest(TABLES);
    if (!container) return;
    var boxes = visibleCheckboxes(container);
    var idx = boxes.indexOf(cb);
    if (idx < 0) return;
    var tid = container.id;
    if (e.shiftKey && anchors[tid] != null && anchors[tid] !== idx &&
        anchors[tid] < boxes.length) {
      var want = cb.checked; // state AFTER this click = state for the range
      var lo = Math.min(anchors[tid], idx);
      var hi = Math.max(anchors[tid], idx);
      synthetic = true;
      try {
        for (var i = lo; i <= hi; i++) {
          if (i !== idx && boxes[i].checked !== want) boxes[i].click();
        }
      } finally {
        synthetic = false;
      }
    }
    anchors[tid] = idx;
  }, true);

  // shift-click otherwise drag-selects page text — suppress inside the tables
  document.addEventListener("mousedown", function (e) {
    if (e.shiftKey && e.target && e.target.closest && e.target.closest(TABLES)) {
      e.preventDefault();
    }
  }, true);
})();
