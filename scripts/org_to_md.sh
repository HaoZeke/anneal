#!/bin/bash

# Org to Markdown Exporter
# Converts an Org-mode file to Markdown format using Emacs in batch mode.
#
# Usage:
#   ./org_to_md.sh <input-org-file> [output-markdown-file]
#
# If the output file is not provided, the script uses the input file's name with a .md extension.
#
# Author: Rohit Goswami (HaoZeke)
# License: MIT
# Website: https://rgoswami.me

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <input-org-file> [output-markdown-file]"
    exit 1
fi

# Input Org file
ORG_FILE=$1

# Output Markdown file
MD_FILE=${2:-"${ORG_FILE%.org}.md"}

# Check if the output file name is different from the input file name
if [ "${ORG_FILE%.org}.md" != "$MD_FILE" ]; then
    # Temporary Org file with the final Markdown filename
    TEMP_ORG_FILE="${MD_FILE%.md}.org"

    # Copy original Org file to temporary file
    cp "$ORG_FILE" "$TEMP_ORG_FILE"

    # Use the temporary file for exporting
    EXPORT_FILE="$TEMP_ORG_FILE"
else
    # Use the original file for exporting
    EXPORT_FILE="$ORG_FILE"
fi

# Emacs command to export Org file to Markdown
# Source blocks become fenced blocks with their language, so the README
# keeps highlighting and docs/doccheck.py can compare them with the Org file.
emacs --batch "$EXPORT_FILE" \
    --eval "(require 'ox-md nil t)" \
    --eval "(advice-add 'org-md-example-block :around
              (lambda (orig block contents info)
                (if (eq (org-element-type block) 'src-block)
                    (format \"\`\`\`%s\n%s\`\`\`\"
                            (or (org-element-property :language block) \"\")
                            (org-remove-indentation
                             (org-export-format-code-default block info)))
                  (funcall orig block contents info))))" \
    --eval "(org-md-export-to-markdown)" \
    --kill

# Cleanup: remove temporary Org file if it was created
if [ -f "$TEMP_ORG_FILE" ]; then
    rm "$TEMP_ORG_FILE"
fi

# Check if Markdown file was created
if [ -f "$MD_FILE" ]; then
    # ox-md ends the file with a blank line, which the end-of-file-fixer hook rejects.
    printf '%s\n' "$(cat "$MD_FILE")" > "$MD_FILE.tmp" && mv "$MD_FILE.tmp" "$MD_FILE"
    echo "Exported to $MD_FILE"
else
    echo "Export failed."
fi
