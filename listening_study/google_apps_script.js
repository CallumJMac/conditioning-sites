/**
 * Google Apps Script for collecting paired-comparison listening study results.
 *
 * Deploy as a Web App:
 *   1. Paste this into Extensions > Apps Script in a Google Sheet
 *   2. Deploy > New deployment > Web app
 *   3. Execute as: Me | Access: Anyone
 *   4. Copy the URL into configs/study.json "resultsEndpoint"
 *
 * The client posts with Content-Type text/plain to avoid a CORS preflight.
 *
 * Read access for monitoring: set a Script property READ_KEY (Project Settings >
 * Script properties). GET <url>?key=<READ_KEY> returns the Raw sheet as JSON;
 * without the key, GET only reports that the endpoint is active.
 */

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const data = JSON.parse(e.postData.contents);
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const now = new Date().toISOString();

    // Raw JSON sheet (one row per submission); analysis reads this.
    const rawSheet = ss.getSheetByName("Raw") || ss.insertSheet("Raw");
    if (rawSheet.getLastRow() === 0) {
      rawSheet.appendRow(["timestamp", "participant_id", "list", "screened_out", "duration_ms", "json"]);
    }
    rawSheet.appendRow([now, data.participantId, data.list, data.screenedOut, data.durationMs, JSON.stringify(data)]);

    // Ratings sheet (one row per trial) for quick inspection.
    const ratingsSheet = ss.getSheetByName("Ratings") || ss.insertSheet("Ratings");
    if (ratingsSheet.getLastRow() === 0) {
      ratingsSheet.appendRow(["timestamp", "participant_id", "list", "axis", "kind", "pair_or_trap", "a", "b", "score", "response_ms"]);
    }
    const rows = (data.responses || []).map(r => [
      now, data.participantId, data.list, r.axis, r.kind, r.pairId || r.trapId, r.a, r.b, r.score, r.responseMs
    ]);
    if (rows.length) {
      ratingsSheet.getRange(ratingsSheet.getLastRow() + 1, 1, rows.length, rows[0].length).setValues(rows);
    }

    return ContentService.createTextOutput(JSON.stringify({ status: "ok" }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ status: "error", message: err.toString() }))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  const readKey = PropertiesService.getScriptProperties().getProperty("READ_KEY");
  if (!readKey || !e || !e.parameter || e.parameter.key !== readKey) {
    return ContentService.createTextOutput("Listening study results endpoint active.");
  }
  const rawSheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Raw");
  const rows = rawSheet ? rawSheet.getDataRange().getValues() : [];
  return ContentService.createTextOutput(JSON.stringify({ status: "ok", rows: rows }))
    .setMimeType(ContentService.MimeType.JSON);
}
