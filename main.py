"""
Entry point for the CRN-RL-Framework.
"""

import sys
import os
import random
import yaml
import numpy as np
import torch
import sys

def main():
    # Route to Aditya's CLI flow
    from cli.parser import get_parser
    from cli.runner import execute_cli
    parser = get_parser()
    args = parser.parse_args()
    execute_cli(args)

if __name__ == "__main__":
    main()
