import os
import sys
import argparse
from http.server import SimpleHTTPRequestHandler
import socketserver

def serve_visualizer(port=8080):
    vis_dir = os.path.join(os.path.dirname(__file__), "visualizer")
    os.chdir(vis_dir)
    
    Handler = SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving CRN Visualizer at http://localhost:{port}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down visualizer server.")

def kaggle_iframe():
    print("Run the following code in a Kaggle Notebook cell to view the visualizer:")
    print('----------------------------------------------------')
    print('from IPython.display import IFrame, HTML')
    print('HTML("<a href=\'visualizer/index.html\' target=\'_blank\'>Click here to open the visualizer in a new tab</a>")')
    print('----------------------------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRN Visualizer Utility")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve the visualizer on")
    parser.add_argument("--kaggle", action="store_true", help="Print instructions for viewing on Kaggle")
    
    args = parser.parse_args()
    
    if args.kaggle:
        kaggle_iframe()
    else:
        serve_visualizer(args.port)
