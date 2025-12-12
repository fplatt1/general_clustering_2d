import streamlit.web.cli as stcli
import os
import sys
import shutil
import tempfile
import multiprocessing

def setup_environment():
    """
    Kopiert die App-Dateien aus der EXE in einen echten temporären Ordner,
    um Konflikte mit Numpy/Pfaden zu vermeiden.
    """
    if getattr(sys, 'frozen', False):
        # 1. Quell-Pfad (innerhalb der EXE)
        # Wir erwarten den Code im Unterordner 'app_content' der EXE
        base_dir = os.path.join(sys._MEIPASS, "app_content") # type: ignore
        
        # 2. Ziel-Pfad: erstelle pro-Ausführung einen eindeutigen Temp-Ordner
        # Dadurch verhindern wir Race-Conditions, wenn mehrere Prozesse parallel laufen.
        temp_dir = tempfile.mkdtemp(prefix="Freddies_Bachelor_App_")

        # 3. Kopieren (ohne große `data`-Ordner). Kopiere nur Quellcode und kleine Assets,
        # damit das Kopieren schnell ist und keine Dateien von anderen Prozessen gesperrt werden.
        try:
            shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('data', 'data/*'))
        except Exception as e:
            # Bei Zugriffsfehlern (z.B. WinError 32) weiterlaufen — wir können in vielen Fällen
            # direkt aus dem Bundle lesen oder den fehlenden Inhalt lazily nachladen.
            print(f"Warnung beim Kopieren: {e}")

        # 4. Arbeitsverzeichnis ändern
        os.chdir(temp_dir)
        
        # 5. Pfade anpassen, damit Python Bibliotheken findet
        sys.path.insert(0, temp_dir)
        
        return os.path.join(temp_dir, "app.py")
    
    else:
        # Normaler Modus (VS Code)
        return os.path.join(os.path.dirname(__file__), "app.py")

if __name__ == "__main__":
    # Umgebung vorbereiten
    app_path = setup_environment()
    
    # Präventive Umgebungsvariablen: Deaktiviere File-Watcher und automatisches Browser-Öffnen
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")
    # Setze BROWSER auf einen ungültigen Wert, damit `webbrowser` kein neues Fenster öffnet
    os.environ.setdefault("BROWSER", "false")

    # Streamlit starten (mit zusätzlichen Flags, die Reloads verhindern)
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
        "--server.headless=true",
    ]
    # Für eingefrorene Executables: unterstütze multiprocessing-Freeze (Windows)
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass

    sys.exit(stcli.main())