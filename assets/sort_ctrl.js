/* Records the ctrl-key state of clicks inside the conversation tables into the
 * hidden #at-sort-ctrl-input BEFORE DataTable processes the click, so the
 * server-side sort-semantics callback can distinguish a plain header click
 * (replace the sort) from a ctrl-click (stack a 2nd/3rd-order sort).
 */
(function () {
  "use strict";
  var TABLES = "#at-explorer-conv-table, #at-deep-conv-table";
  var setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, "value").set;

  document.addEventListener("mousedown", function (e) {
    if (!(e.target && e.target.closest && e.target.closest(TABLES))) return;
    var inp = document.getElementById("at-sort-ctrl-input");
    if (!inp) return;
    var v = e.ctrlKey ? "1" : "0";
    if (inp.value === v) return; // no change -> no event churn
    setter.call(inp, v);
    inp.dispatchEvent(new Event("input", { bubbles: true }));
  }, true);
})();
