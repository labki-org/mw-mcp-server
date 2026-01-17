"""
Usage Statistics and Admin Dashboard

This module provides endpoints for tracking usage of the MCP server
and a simple HTML dashboard for visualization.

Security
--------
All endpoints are protected by `verify_admin` which requires:
- `x-admin-key` header OR
- `key` query parameter

Metrics Tracked
---------------
- Token Usage (Input/Output/Total)
- Chat Sessions Created
- Messages Exchanged
- Active Users
- Embeddings Count (Snapshot)
"""

import uuid
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Header, status
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from sqlalchemy import func, select, desc
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
    key: Optional[str] = Query(None)
):
    """
    Verify the request is from an admin using the configured API key.
    Checks header first, then query param.
    """
    expected_key = settings.admin_api_key.get_secret_value() if settings.admin_api_key else None
    
    if not expected_key:
        # If no key is configured, disable admin access securely
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is not configured (ADMIN_API_KEY missing)"
        )

    provided_key = x_admin_key or key
    
    if not provided_key or provided_key != expected_key:
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

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui(
    x_admin_key: Optional[str] = Header(None, alias="x-admin-key"),
    key: Optional[str] = Query(None)
):
    """
    Serve the admin dashboard HTML.
    Note: We do NOT use verify_admin dependency here because we want to serve
    the page even if auth is missing, so the JS can prompt for it.
    The data endpoints ARE protected.
    """
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MW MCP Admin Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
        .card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 1.5rem; }
    </style>
</head>
<body class="p-6">

    <!-- Auth Modal -->
    <div id="authModal" class="fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center hidden z-50">
        <div class="bg-white p-6 rounded-lg shadow-xl w-96">
            <h2 class="text-xl font-bold mb-4">Admin Authentication</h2>
            <p class="text-gray-600 mb-4 text-sm">Please enter your Admin API Key to continue.</p>
            <input type="password" id="apiKeyInput" class="w-full border p-2 rounded mb-4" placeholder="Enter API Key">
            <button onclick="saveKeyAndReload()" class="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700">Access Dashboard</button>
        </div>
    </div>

    <!-- Header -->
    <div class="max-w-7xl mx-auto mb-8 flex justify-between items-center">
        <div>
            <h1 class="text-3xl font-bold text-gray-800">Usage Statistics</h1>
            <p class="text-gray-500 text-sm mt-1">Admin Dashboard</p>
        </div>
        <div class="flex gap-4">
            <select id="timeRange" onchange="fetchData()" class="border p-2 rounded bg-white shadow-sm">
                <option value="7">Last 7 Days</option>
                <option value="30" selected>Last 30 Days</option>
                <option value="90">Last 90 Days</option>
            </select>
            <button onclick="logout()" class="text-red-600 text-sm hover:underline">Logout</button>
        </div>
    </div>

    <!-- Dashboard Grid -->
    <div class="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <!-- Summary Cards -->
        <div class="card border-l-4 border-blue-500">
            <h3 class="text-gray-500 text-sm uppercase font-semibold">Total Tokens</h3>
            <p class="text-3xl font-bold text-gray-800 mt-2" id="totalTokens">-</p>
        </div>
        <div class="card border-l-4 border-green-500">
            <h3 class="text-gray-500 text-sm uppercase font-semibold">Active Users</h3>
            <p class="text-3xl font-bold text-gray-800 mt-2" id="totalUsers">-</p>
        </div>
        <div class="card border-l-4 border-purple-500">
            <h3 class="text-gray-500 text-sm uppercase font-semibold">Chat Sessions</h3>
            <p class="text-3xl font-bold text-gray-800 mt-2" id="totalSessions">-</p>
        </div>
        <div class="card border-l-4 border-orange-500">
            <h3 class="text-gray-500 text-sm uppercase font-semibold">Total Embeddings</h3>
            <p class="text-3xl font-bold text-gray-800 mt-2" id="totalEmbeddings">-</p>
        </div>
    </div>

    <!-- Charts Area -->
    <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div class="card">
            <h3 class="font-bold text-gray-700 mb-4">Token Usage by Tenant</h3>
            <canvas id="tokensChart"></canvas>
        </div>
        <div class="card">
            <h3 class="font-bold text-gray-700 mb-4">Activity Breakdown</h3>
            <canvas id="activityChart"></canvas>
        </div>
    </div>

    <!-- Data Table -->
    <div class="max-w-7xl mx-auto card overflow-hidden">
        <h3 class="font-bold text-gray-700 mb-4">Tenant Details</h3>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="border-b bg-gray-50 text-gray-600 text-sm uppercase">
                        <th class="p-3">Tenant (Wiki ID)</th>
                        <th class="p-3 text-right">Tokens</th>
                        <th class="p-3 text-right">Sessions</th>
                        <th class="p-3 text-right">Messages</th>
                        <th class="p-3 text-right">Users</th>
                        <th class="p-3 text-right">Embeddings</th>
                    </tr>
                </thead>
                <tbody id="statsTableBody" class="text-gray-700 text-sm">
                    <!-- Rows injected by JS -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const API_BASE = '/stats/usage';
        let charts = {};

        // --- Auth Logic ---
        function getApiKey() {
            // Check URL param first
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('key')) return urlParams.get('key');
            // Check LocalStorage
            return localStorage.getItem('admin_api_key');
        }

        function checkAuth() {
            const key = getApiKey();
            if (!key) {
                document.getElementById('authModal').classList.remove('hidden');
                return false;
            }
            return true;
        }

        function saveKeyAndReload() {
            const key = document.getElementById('apiKeyInput').value;
            if (key) {
                localStorage.setItem('admin_api_key', key);
                // Reload without query param if possible to clean URL, or just reload
                window.location.reload();
            }
        }

        function logout() {
            localStorage.removeItem('admin_api_key');
            window.location.href = window.location.pathname; // strip query params
        }

        // --- Data Fetching ---
        async function fetchData() {
            if (!checkAuth()) return;
            
            const days = document.getElementById('timeRange').value;
            const key = getApiKey();

            try {
                const response = await fetch(`${API_BASE}?days=${days}`, {
                    headers: { 'x-admin-key': key }
                });

                if (response.status === 403) {
                    alert('Invalid API Key. Please logout and try again.');
                    logout();
                    return;
                }
                
                if (!response.ok) throw new Error('Failed to fetch data');

                const data = await response.json();
                renderDashboard(data);
                
            } catch (err) {
                console.error(err);
                alert('Error loading dashboard data');
            }
        }

        // --- Rendering ---
        function renderDashboard(data) {
            const tenants = data.tenants;

            // Update Summary Cards
            let totalTokens = 0, totalUsers = 0, totalSessions = 0, totalEmbeddings = 0;
            tenants.forEach(t => {
                totalTokens += t.total_tokens;
                totalUsers += t.active_users; // Note: this is sum of distinct users PER tenant, so global unique might be slightly lower if users cross wikis, but good enough.
                totalSessions += t.session_count;
                totalEmbeddings += t.embedding_count;
            });

            document.getElementById('totalTokens').innerText = totalTokens.toLocaleString();
            document.getElementById('totalUsers').innerText = totalUsers.toLocaleString();
            document.getElementById('totalSessions').innerText = totalSessions.toLocaleString();
            document.getElementById('totalEmbeddings').innerText = totalEmbeddings.toLocaleString();

            // Render Charts
            renderCharts(tenants);

            // Render Table
            const tbody = document.getElementById('statsTableBody');
            tbody.innerHTML = tenants.map(t => `
                <tr class="border-b hover:bg-gray-50">
                    <td class="p-3 font-medium">${t.wiki_id}</td>
                    <td class="p-3 text-right">${t.total_tokens.toLocaleString()}</td>
                    <td class="p-3 text-right">${t.session_count.toLocaleString()}</td>
                    <td class="p-3 text-right">${t.message_count.toLocaleString()}</td>
                    <td class="p-3 text-right">${t.active_users.toLocaleString()}</td>
                    <td class="p-3 text-right">${t.embedding_count.toLocaleString()}</td>
                </tr>
            `).join('');
        }

        function renderCharts(tenants) {
            const labels = tenants.map(t => t.wiki_id);
            
            // 1. Token Chart
            const ctxTokens = document.getElementById('tokensChart').getContext('2d');
            if (charts.tokens) charts.tokens.destroy();
            
            charts.tokens = new Chart(ctxTokens, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Prompt Tokens',
                            data: tenants.map(t => t.prompt_tokens),
                            backgroundColor: '#60A5FA'
                        },
                        {
                            label: 'Completion Tokens',
                            data: tenants.map(t => t.completion_tokens),
                            backgroundColor: '#3B82F6'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { stacked: true },
                        y: { stacked: true }
                    }
                }
            });

            // 2. Activity Chart
            const ctxActivity = document.getElementById('activityChart').getContext('2d');
            if (charts.activity) charts.activity.destroy();

            charts.activity = new Chart(ctxActivity, {
                type: 'bar', // grouped bar
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Sessions',
                            data: tenants.map(t => t.session_count),
                            backgroundColor: '#A855F7'
                        },
                        {
                            label: 'Messages',
                            data: tenants.map(t => t.message_count),
                            backgroundColor: '#F97316'
                        }
                    ]
                },
                options: {
                    responsive: true
                }
            });
        }

        // Init
        document.addEventListener('DOMContentLoaded', () => {
             // Check if we need to auto-login from URL
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('key')) {
                localStorage.setItem('admin_api_key', urlParams.get('key'));
            }
            
            if (checkAuth()) {
                fetchData();
            }
        });

    </script>
</body>
</html>
    """
    return html_content
