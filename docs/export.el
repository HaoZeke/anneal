;;; export.el --- Export orgmode docs to RST for Sphinx -*- lexical-binding: t -*-

;; Usage: emacs --batch -l docs/export.el
;; Aligns with rsx-rs radsex-export-all recursive style for maintainability;
;; keeps recursive export and ox-rst. See rgpycrumbs/docs/export.el for pandoc
;; pre-pass variant.

;; Install ox-rst from MELPA if not already available
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)
(unless (package-installed-p 'ox-rst)
  (package-refresh-contents)
  (package-install 'ox-rst))

(require 'org)
(require 'ox-rst)

(defun radsex-export-all ()
  "Export all orgmode files under docs/orgmode/ to RST under docs/source/.
Recursive by construction (directory-files-recursively). Matches the
Diataxis layout (index, quickstart, tutorials/*, howto/*, explanation/*,
reference/*, dev/*) and preserves existing publish :recursive t behaviour
while providing explicit per-file messages."
  (let* ((script-dir (file-name-directory (or load-file-name buffer-file-name)))
         (org-dir (expand-file-name "orgmode" script-dir))
         (rst-dir (expand-file-name "source" script-dir)))
    (dolist (rst-file (directory-files-recursively rst-dir "\\.rst$"))
      (delete-file rst-file))
    (dolist (org-file (directory-files-recursively org-dir "\\.org$"))
      (let* ((relative (file-relative-name org-file org-dir))
             (rst-file (expand-file-name
                        (concat (file-name-sans-extension relative) ".rst")
                        rst-dir))
             (rst-parent (file-name-directory rst-file)))
        (unless (file-directory-p rst-parent)
          (make-directory rst-parent t))
        (with-current-buffer (find-file-noselect org-file)
          (org-export-to-file 'rst rst-file nil nil nil nil)
          (kill-buffer))
        (message "Exported %s -> %s" org-file rst-file)))))
(radsex-export-all)
