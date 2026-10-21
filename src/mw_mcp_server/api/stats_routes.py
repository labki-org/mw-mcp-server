"""
Usage Statistics and Admin Dashboard

This module provides endpoints for tracking usage of the MCP server
and a simple HTML dashboard for visualization.

Security
--------
Data endpoints are protected by `verify_admin`, which requires the
`x-admin-key` header. Query-string keys are deliberately NOT accepted
(they leak into access logs, browser history and Referer headers).

Metrics Tracked
---------------
- Token Usage (Input/Output/Total)
- Chat Sessions Created
- Messages Exchanged
- Active Users
- Embeddings Count (Snapshot)
"""

import hmac
import secrets
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Header, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, ConfigDict

from ..config import settings
from ..db import get_async_session
from ..db.models import TokenUsage, ChatSession, ChatMessage, Embedding

router = APIRouter(prefix="/stats", tags=["stats"])


# ---------------------------------------------------------------------
# Security Dependency
# ---------------------------------------------------------------------

async def verify_admin(
    x_admin_key: Optional[str] = Header(None, alias="x-admin-key"),
):
    """
    Verify the request is from an admin using the configured API key,
    supplied via the ``x-admin-key`` header only.
    """
    expected_key = settings.admin_api_key.get_secret_value() if settings.admin_api_key else None
    
    if not expected_key:
        # If no key is configured, disable admin access securely
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is not configured (ADMIN_API_KEY missing)"
        )

    provided_key = x_admin_key

    if not provided_key or not hmac.compare_digest(
        provided_key.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing admin API key"
        )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class TenantStats(BaseModel):
    wiki_id: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    request_count: int
    session_count: int
    message_count: int
    active_users: int
    embedding_count: int
    
    model_config = ConfigDict(from_attributes=True)

class GlobalStats(BaseModel):
    period: str
    start_date: date
    end_date: date
    tenants: List[TenantStats]


# ---------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------

@router.get("/usage", response_model=GlobalStats, dependencies=[Depends(verify_admin)])
async def get_usage_stats(
    period: str = Query("day", pattern="^(day|week|month|all)$"),
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get usage statistics aggregated by tenant.
    
    Parameters
    ----------
    period : str
        Aggregation period (currently effectively validates window).
    days : int
        Number of days to look back.
    """
    start_date = date.today() - timedelta(days=days)
    end_date = date.today()
    
    # 1. Token Usage Aggregation
    #    Group by wiki_id
    token_query = (
        select(
            TokenUsage.wiki_id,
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.sum(TokenUsage.prompt_tokens).label("prompt_tokens"),
            func.sum(TokenUsage.completion_tokens).label("completion_tokens"),
            func.sum(TokenUsage.request_count).label("request_count"),
            func.count(func.distinct(TokenUsage.user_id)).label("active_users"),
        )
        .where(TokenUsage.usage_date >= start_date)
        .group_by(TokenUsage.wiki_id)
    )
    token_result = await db.execute(token_query)
    token_map = {
        row.wiki_id: {
            "total_tokens": row.total_tokens or 0,
            "prompt_tokens": row.prompt_tokens or 0,
            "completion_tokens": row.completion_tokens or 0,
            "request_count": row.request_count or 0,
            "active_users": row.active_users or 0
        }
        for row in token_result
    }

    # 2. Session Count
    session_query = (
        select(
            ChatSession.wiki_id,
            func.count(ChatSession.session_id).label("session_count")
        )
        .where(ChatSession.created_at >= datetime.combine(start_date, datetime.min.time()))
        .group_by(ChatSession.wiki_id)
    )
    session_result = await db.execute(session_query)
    session_map = {row.wiki_id: row.session_count for row in session_result}

    # 3. Message Count
    #    Need to join session to get wiki_id
    message_query = (
        select(
            ChatSession.wiki_id,
            func.count(ChatMessage.message_id).label("message_count")
        )
        .join(ChatMessage.session)
        .where(ChatMessage.created_at >= datetime.combine(start_date, datetime.min.time()))
        .group_by(ChatSession.wiki_id)
    )
    message_result = await db.execute(message_query)
    message_map = {row.wiki_id: row.message_count for row in message_result}
    
    # 4. Embedding Count (Snapshot - Current Total)
    embedding_query = (
        select(
            Embedding.wiki_id,
            func.count(Embedding.id).label("embedding_count")
        )
        .group_by(Embedding.wiki_id)
    )
    embedding_result = await db.execute(embedding_query)
    embedding_map = {row.wiki_id: row.embedding_count for row in embedding_result}

    # Merge all data
    all_wikis = set(token_map.keys()) | set(session_map.keys()) | set(message_map.keys()) | set(embedding_map.keys())
    
    tenant_stats = []
    for wiki_id in sorted(all_wikis):
        t_stats = token_map.get(wiki_id, {})
        stats = TenantStats(
            wiki_id=wiki_id,
            total_tokens=t_stats.get("total_tokens", 0),
            prompt_tokens=t_stats.get("prompt_tokens", 0),
            completion_tokens=t_stats.get("completion_tokens", 0),
            request_count=t_stats.get("request_count", 0),
            active_users=t_stats.get("active_users", 0),
            session_count=session_map.get(wiki_id, 0),
            message_count=message_map.get(wiki_id, 0),
            embedding_count=embedding_map.get(wiki_id, 0),
        )
        tenant_stats.append(stats)

    return GlobalStats(
        period=period,
        start_date=start_date,
        end_date=end_date,
        tenants=tenant_stats
    )


# ---------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------

# Chart.js is loaded from a CDN pinned to an exact version with Subresource
# Integrity. To bump: download the exact file and recompute
#   curl -sL <url> | openssl dgst -sha384 -binary | openssl base64 -A
CHARTJS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js"
CHARTJS_SRI = "sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ"


def _dashboard_csp(nonce: str) -> str:
    return "; ".join(
        [
            "default-src 'none'",
            f"script-src 'nonce-{nonce}' {CHARTJS_URL}",
            f"style-src 'nonce-{nonce}'",
            "connect-src 'self'",
            "img-src 'self' data:",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
        ]
    )


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="referrer" content="no-referrer">
<title>MW MCP Admin Dashboard</title>
<script nonce="__NONCE__" src="__CHARTJS_URL__" integrity="__CHARTJS_SRI__" crossorigin="anonymous"></script>
<style nonce="__NONCE__">
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; padding: 1.5rem; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         background: #f3f4f6; color: #1f2937; }
  .wrap { max-width: 80rem; margin: 0 auto 2rem; }
  .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
  h1 { font-size: 1.875rem; margin: 0; }
  .muted { color: #6b7280; font-size: .875rem; margin: .25rem 0 0; }
  .controls { display: flex; gap: 1rem; align-items: center; }
  select, input { border: 1px solid #d1d5db; border-radius: .375rem; padding: .5rem; background: #fff; font: inherit; }
  button { font: inherit; cursor: pointer; border: 0; border-radius: .375rem; padding: .5rem 1rem; }
  .btn-primary { background: #2563eb; color: #fff; width: 100%; }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-link { background: none; color: #dc2626; padding: 0; font-size: .875rem; }
  .grid { display: grid; gap: 1.5rem; grid-template-columns: 1fr; }
  @media (min-width: 768px) { .grid-4 { grid-template-columns: repeat(2, 1fr); } }
  @media (min-width: 1024px) { .grid-4 { grid-template-columns: repeat(4, 1fr); }
                               .grid-2 { grid-template-columns: repeat(2, 1fr); } }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.1); padding: 1.5rem; }
  .accent-blue { border-left: 4px solid #3b82f6; } .accent-green { border-left: 4px solid #22c55e; }
  .accent-purple { border-left: 4px solid #a855f7; } .accent-orange { border-left: 4px solid #f97316; }
  .card h3 { margin: 0 0 1rem; font-size: 1rem; color: #374151; }
  .stat-label { margin: 0; color: #6b7280; font-size: .875rem; text-transform: uppercase; font-weight: 600; }
  .stat-value { margin: .5rem 0 0; font-size: 1.875rem; font-weight: 700; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: .875rem; text-align: left; }
  th { background: #f9fafb; color: #4b5563; text-transform: uppercase; }
  th, td { padding: .75rem; border-bottom: 1px solid #e5e7eb; }
  .num { text-align: right; }
  .modal { position: fixed; inset: 0; background: rgba(17,24,39,.5); display: flex;
           align-items: center; justify-content: center; z-index: 50; }
  .modal-box { background: #fff; padding: 1.5rem; border-radius: .5rem; width: 24rem; max-width: 100%; }
  .modal-box input { width: 100%; margin-bottom: 1rem; }
  .error { color: #dc2626; font-size: .875rem; min-height: 1.25rem; margin: 0 0 .5rem; }
  .hidden { display: none; }
</style>
</head>
<body>

<div id="authModal" class="modal hidden">
  <form id="authForm" class="modal-box">
    <h2>Admin Authentication</h2>
    <p class="muted">Enter the Admin API key. It is kept only for this browser tab.</p>
    <input type="password" id="apiKeyInput" placeholder="Admin API key" autocomplete="off" required>
    <p id="authError" class="error"></p>
    <button type="submit" class="btn-primary">Access Dashboard</button>
  </form>
</div>

<div class="wrap header">
  <div>
    <h1>Usage Statistics</h1>
    <p class="muted">Admin Dashboard</p>
  </div>
  <div class="controls">
    <select id="timeRange">
      <option value="7">Last 7 Days</option>
      <option value="30" selected>Last 30 Days</option>
      <option value="90">Last 90 Days</option>
    </select>
    <button id="logoutBtn" type="button" class="btn-link">Logout</button>
  </div>
</div>

<div class="wrap grid grid-4">
  <div class="card accent-blue"><p class="stat-label">Total Tokens</p><p class="stat-value" id="totalTokens">-</p></div>
  <div class="card accent-green"><p class="stat-label">Active Users</p><p class="stat-value" id="totalUsers">-</p></div>
  <div class="card accent-purple"><p class="stat-label">Chat Sessions</p><p class="stat-value" id="totalSessions">-</p></div>
  <div class="card accent-orange"><p class="stat-label">Total Embeddings</p><p class="stat-value" id="totalEmbeddings">-</p></div>
</div>

<div class="wrap grid grid-2">
  <div class="card"><h3>Token Usage by Tenant</h3><canvas id="tokensChart"></canvas></div>
  <div class="card"><h3>Activity Breakdown</h3><canvas id="activityChart"></canvas></div>
</div>

<div class="wrap card">
  <h3>Tenant Details</h3>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Tenant (Wiki ID)</th><th class="num">Tokens</th><th class="num">Sessions</th>
          <th class="num">Messages</th><th class="num">Users</th><th class="num">Embeddings</th>
        </tr>
      </thead>
      <tbody id="statsTableBody"></tbody>
    </table>
  </div>
</div>

<script nonce="__NONCE__">
(function () {
  'use strict';
  const API_BASE = '/stats/usage';
  const STORAGE_KEY = 'mcp_admin_api_key';
  const charts = {};
  let apiKey = null;

  // The key is sent only in the x-admin-key header. It is held in memory and
  // mirrored to sessionStorage (cleared when the tab closes), never in
  // localStorage or the URL.
  function loadKey() {
    try { apiKey = sessionStorage.getItem(STORAGE_KEY); } catch (e) { apiKey = null; }
    return apiKey;
  }
  function storeKey(key) {
    apiKey = key;
    try { sessionStorage.setItem(STORAGE_KEY, key); } catch (e) { /* memory only */ }
  }
  function clearKey() {
    apiKey = null;
    try { sessionStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
  }

  function showAuth(message) {
    document.getElementById('authError').textContent = message || '';
    document.getElementById('authModal').classList.remove('hidden');
    document.getElementById('apiKeyInput').focus();
  }
  function hideAuth() {
    document.getElementById('authModal').classList.add('hidden');
    document.getElementById('apiKeyInput').value = '';
  }

  async function fetchData() {
    if (!apiKey) { showAuth(); return; }
    const days = encodeURIComponent(document.getElementById('timeRange').value);
    try {
      const response = await fetch(API_BASE + '?days=' + days, {
        headers: { 'x-admin-key': apiKey },
        cache: 'no-store',
        credentials: 'omit',
      });
      if (response.status === 403) {
        clearKey();
        showAuth('Invalid API key.');
        return;
      }
      if (!response.ok) throw new Error('HTTP ' + response.status);
      renderDashboard(await response.json());
    } catch (err) {
      console.error(err);
      alert('Error loading dashboard data');
    }
  }

  function fmt(n) { return Number(n || 0).toLocaleString(); }

  function cell(text, numeric) {
    const td = document.createElement('td');
    td.textContent = text;  // textContent: tenant IDs are never parsed as HTML
    if (numeric) td.className = 'num';
    return td;
  }

  function renderDashboard(data) {
    const tenants = data.tenants || [];
    let totalTokens = 0, totalUsers = 0, totalSessions = 0, totalEmbeddings = 0;
    tenants.forEach(function (t) {
      totalTokens += t.total_tokens;
      totalUsers += t.active_users;  // per-tenant distinct users, summed
      totalSessions += t.session_count;
      totalEmbeddings += t.embedding_count;
    });
    document.getElementById('totalTokens').textContent = fmt(totalTokens);
    document.getElementById('totalUsers').textContent = fmt(totalUsers);
    document.getElementById('totalSessions').textContent = fmt(totalSessions);
    document.getElementById('totalEmbeddings').textContent = fmt(totalEmbeddings);

    renderCharts(tenants);

    const tbody = document.getElementById('statsTableBody');
    tbody.replaceChildren();
    tenants.forEach(function (t) {
      const tr = document.createElement('tr');
      tr.append(
        cell(String(t.wiki_id), false),
        cell(fmt(t.total_tokens), true),
        cell(fmt(t.session_count), true),
        cell(fmt(t.message_count), true),
        cell(fmt(t.active_users), true),
        cell(fmt(t.embedding_count), true)
      );
      tbody.appendChild(tr);
    });
  }

  function renderCharts(tenants) {
    if (typeof Chart === 'undefined') return;
    const labels = tenants.map(function (t) { return String(t.wiki_id); });

    if (charts.tokens) charts.tokens.destroy();
    charts.tokens = new Chart(document.getElementById('tokensChart'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Prompt Tokens', data: tenants.map(function (t) { return t.prompt_tokens; }), backgroundColor: '#60A5FA' },
          { label: 'Completion Tokens', data: tenants.map(function (t) { return t.completion_tokens; }), backgroundColor: '#3B82F6' }
        ]
      },
      options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true } } }
    });

    if (charts.activity) charts.activity.destroy();
    charts.activity = new Chart(document.getElementById('activityChart'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Sessions', data: tenants.map(function (t) { return t.session_count; }), backgroundColor: '#A855F7' },
          { label: 'Messages', data: tenants.map(function (t) { return t.message_count; }), backgroundColor: '#F97316' }
        ]
      },
      options: { responsive: true }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('authForm').addEventListener('submit', function (ev) {
      ev.preventDefault();
      const key = document.getElementById('apiKeyInput').value;
      if (!key) return;
      storeKey(key);
      hideAuth();
      fetchData();
    });
    document.getElementById('timeRange').addEventListener('change', fetchData);
    document.getElementById('logoutBtn').addEventListener('click', function () {
      clearKey();
      showAuth();
    });
    if (loadKey()) { fetchData(); } else { showAuth(); }
  });
})();
</script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui() -> HTMLResponse:
    """
    Serve the admin dashboard HTML.

    The page itself is static and contains no data, so it is served without
    auth; the JS prompts for the admin key and sends it in the
    ``x-admin-key`` header to the protected ``/stats/usage`` endpoint.
    A strict nonce-based Content-Security-Policy is attached.
    """
    nonce = secrets.token_urlsafe(16)
    html = (
        _DASHBOARD_HTML.replace("__NONCE__", nonce)
        .replace("__CHARTJS_URL__", CHARTJS_URL)
        .replace("__CHARTJS_SRI__", CHARTJS_SRI)
    )
    return HTMLResponse(
        content=html,
        headers={
            "Content-Security-Policy": _dashboard_csp(nonce),
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        },
    )
