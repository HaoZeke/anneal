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

(setq user-full-name "anneal developers")
(require 'org)
;; ox-rst 20250428 is byte-compiled against Org's inlined
;; `org-element-property'. For a standard property such as `:value`
;; that inline expands to `org-element-property-raw' and never resolves
;; an `org-element-deferred' value. `string-match' then receives the
;; deferred vector. Loading the source, with native compilation off,
;; keeps the calls as function calls, and `org-element--property'
;; resolves them.
(setq inhibit-automatic-native-compilation t)
(require 'ox-rst)
(load (locate-library "ox-rst.el") nil nil t)

(defun radsex-changelog-body (org-dir)
  "Write the reStructuredText form of CHANGELOG.md beside the page.
An `export markdown' include is dropped by ox-rst, and an Org
conversion of the same file turns the towncrier marker into an
unknown directive. The page includes this file as an RST export
block, which ox-rst copies through."
  (let ((markdown (expand-file-name "../../CHANGELOG.md" org-dir))
        (body (expand-file-name ".changelog-body.rst" org-dir)))
    (unless (zerop (call-process "pandoc" nil nil nil
                                 "-f" "markdown+tex_math_single_backslash"
                                 "-t" "rst" markdown "-o" body))
      (error "pandoc could not convert %s" markdown))))

(defun anneal-rewrite-rst-links (rst-file rst-dir repo-root)
  "Rewrite ox-rst external .rst and repo-relative source links in RST-FILE.
Links to sibling .rst files under RST-DIR become Sphinx :doc: roles;
links that walk above RST-DIR into src/, examples/, or tests/ become
GitHub blob links."
  (let ((rst-parent (file-name-directory rst-file))
        (repo-root-dir (file-name-as-directory (expand-file-name repo-root))))
    (with-temp-buffer
      (insert-file-contents rst-file)
      (goto-char (point-min))
      (while (re-search-forward
              "`\\([^`\n]+?\\) <\\([^>\n]+\\)>`__?" nil t)
        (let* ((start (match-beginning 0))
               (end (match-end 0))
               (label (match-string 1))
               (target (match-string 2))
               (replacement
                (cond
                 ((string-match-p "^\\(https?://\\|mailto:\\|#\\)" target)
                  nil)
                 ((string-match "^\\([^#]+\\)\\.rst\\(#.*\\)?$" target)
                  (let* ((path-part (match-string 1 target))
                         (resolved (expand-file-name (concat path-part ".rst") rst-parent))
                         (rel (file-relative-name resolved rst-dir)))
                    (unless (string-prefix-p ".." rel)
                      (format ":doc:`%s </%s>`" label (file-name-sans-extension rel)))))
                 ((string-match "^\\.\\./\\.\\./\\.\\./\\(.+\\)$" target)
                  (let* ((resolved (expand-file-name target rst-parent))
                         (repo-rel (file-relative-name resolved repo-root-dir)))
                    (unless (string-prefix-p ".." repo-rel)
                      (format "`%s <https://github.com/HaoZeke/anneal/blob/main/%s>`__"
                              label repo-rel)))))))
          (when replacement
            (goto-char start)
            (delete-region start end)
            (insert replacement))))
      (write-region (point-min) (point-max) rst-file nil 'quiet))))

(defun radsex-export-all ()
  "Export all orgmode files under docs/orgmode/ to RST under docs/source/.
Recursive by construction (directory-files-recursively). Matches the
Diataxis layout (index, quickstart, tutorials/*, howto/*, explanation/*,
reference/*, dev/*) and preserves existing publish :recursive t behaviour
while providing explicit per-file messages."
  (let* ((script-dir (file-name-directory (or load-file-name buffer-file-name)))
         (repo-root (expand-file-name ".." script-dir))
         (org-dir (expand-file-name "orgmode" script-dir))
         (rst-dir (expand-file-name "source" script-dir)))
    (radsex-changelog-body org-dir)
    (dolist (rst-file (directory-files-recursively rst-dir "\\.rst$"))
      (delete-file rst-file))
    (dolist (org-file (directory-files-recursively org-dir "\\.org$"))
      (unless (string-prefix-p "." (file-name-nondirectory org-file))
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
          (anneal-rewrite-rst-links rst-file rst-dir repo-root)
          (message "Exported %s -> %s" org-file rst-file))))))
(radsex-export-all)
