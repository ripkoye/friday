#!/usr/bin/env python3
"""
Startup script for the Friday voice assistant system
"""
import subprocess
import time
import sys
import os

def start_component(name, command, delay=2):
    """Start a component with error handling"""
    print(f"\n🚀 Starting {name}...")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ {name} started (PID: {process.pid})")
        time.sleep(delay)
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def main():
    """Start all components of the Friday system"""
    print("🎤 Starting Friday Voice Assistant System...")
    print("=" * 50)
    
    # Change to iteration2 directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    processes = []
    
    # Get the virtual environment Python path
    venv_python = "../fridayenv/bin/python"
    
    # Start components in order (with virtual environment)
    components = [
        ("API Server", f"{venv_python} api.py"),
        ("AudioServer (Transcriber)", f"{venv_python} AudioServer.py"),
        ("TextClassifiers", f"{venv_python} TextClassifiers.py"),
        ("Planner", f"{venv_python} planner.py"),
        ("VoiceAgent", f"{venv_python} VoiceAgent.py"),
        ("TTS Server", f"{venv_python} tts_server.py"),
    ]
    
    for name, command in components:
        process = start_component(name, command)
        if process:
            processes.append((name, process))
        else:
            print(f"❌ Failed to start {name}, stopping...")
            break
    
    if len(processes) == len(components):
        print("\n🎉 All components started successfully!")
        print("\n📋 Running components:")
        for name, process in processes:
            print(f"  • {name} (PID: {process.pid})")
        
        print("\n🔗 System endpoints:")
        print("  • API Server: http://localhost:8000")
        print("  • AudioServer: localhost:8080 (TCP)")
        print("  • TTS Server: localhost:8081 (TCP)")
        print("  • Redis: localhost:6379")
        
        print("\n💡 To test the system:")
        print("  1. Run your client to connect to AudioServer (port 8080)")
        print("  2. Say 'Friday' to trigger the system")
        print("  3. Ask questions that might need web search")
        
        print("\n⏹️  Press Ctrl+C to stop all components...")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
                # Check if any process died
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"⚠️  {name} stopped unexpectedly")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down all components...")
            
            for name, process in processes:
                print(f"  Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
                except Exception as e:
                    print(f"  Error stopping {name}: {e}")
            
            print("✅ All components stopped")
    
    else:
        print("❌ Failed to start all components")
        sys.exit(1)

if __name__ == "__main__":
    main()
