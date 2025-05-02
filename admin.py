#!/usr/bin/env python3
"""
Admin CLI tool for the Gemini PDF Chatbot

This script allows administrators to:
1. Upload PDFs for processing
2. View all chat histories
3. Get statistics about the vector database
"""

import argparse
import os
import requests
import json
from rich.console import Console
from rich.table import Table
from datetime import datetime

# API base URL - change as needed
API_BASE_URL = "http://localhost:8000"
console = Console()

def upload_pdfs(pdf_files):
    """Upload PDF files to the admin endpoint"""
    files = [("files", (os.path.basename(f), open(f, "rb"), "application/pdf")) for f in pdf_files]
    
    url = f"{API_BASE_URL}/admin/upload-pdf/"
    
    with console.status(f"Uploading {len(pdf_files)} PDF files..."):
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        console.print(f"[green]Successfully processed {len(result['files'])} PDF files[/green]")
        console.print(f"Total chunks created: {result['chunks']}")
        for file in result['files']:
            console.print(f"  - {file}")
    else:
        console.print(f"[red]Error: {response.status_code} - {response.text}[/red]")

def view_sessions():
    """View all chat sessions"""
    url = f"{API_BASE_URL}/admin/sessions/"
    
    with console.status("Fetching chat sessions..."):
        response = requests.get(url)
    
    if response.status_code == 200:
        sessions = response.json()["sessions"]
        
        # Create a summary table
        table = Table(title="Chat Session Summary")
        table.add_column("Session ID", style="cyan")
        table.add_column("Messages", justify="right")
        table.add_column("Created", style="green")
        table.add_column("Last Activity", style="green")
        table.add_column("First Message", style="yellow")
        
        for session_id, data in sessions.items():
            short_id = session_id[:8] + "..."
            table.add_row(
                short_id,
                str(data["message_count"]),
                data["created_at"].split("T")[0] if "T" in data["created_at"] else data["created_at"],
                data["last_activity"].split("T")[0] if "T" in data["last_activity"] else data["last_activity"],
                data["first_user_message"]
            )
        
        console.print(table)
        
        # Ask if user wants to see details of a specific session
        console.print("\nEnter a session ID to view details (or press Enter to return):")
        session_choice = input("> ")
        
        if session_choice and session_choice in sessions:
            # Show detailed view of the chosen session
            console.print(f"\n[bold]Session: {session_choice}[/bold]\n")
            
            for msg in sessions[session_choice]["history"]:
                role_color = "green" if msg["role"] == "assistant" else "blue"
                console.print(f"[{role_color}]{msg['role'].upper()}:[/{role_color}] {msg['content']}\n")
        
    else:
        console.print(f"[red]Error: {response.status_code} - {response.text}[/red]")

def get_db_stats():
    """Get vector database statistics"""
    url = f"{API_BASE_URL}/admin/vector-db/stats"
    
    with console.status("Fetching vector database statistics..."):
        response = requests.get(url)
    
    if response.status_code == 200:
        stats = response.json()
        
        console.print("[bold]Vector Database Statistics[/bold]")
        console.print(f"Document Count: {stats['document_count']}")
        console.print(f"Index Size: {stats['index_size']} vectors")
    else:
        console.print(f"[red]Error: {response.status_code} - {response.text}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Admin CLI for Gemini PDF Chatbot")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Upload PDFs command
    upload_parser = subparsers.add_parser("upload", help="Upload PDF files")
    upload_parser.add_argument("files", nargs="+", help="PDF files to upload")
    
    # View sessions command
    sessions_parser = subparsers.add_parser("sessions", help="View all chat sessions")
    
    # Vector DB stats command
    stats_parser = subparsers.add_parser("stats", help="Get vector database statistics")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        upload_pdfs(args.files)
    elif args.command == "sessions":
        view_sessions()
    elif args.command == "stats":
        get_db_stats()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()