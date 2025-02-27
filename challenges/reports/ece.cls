% UWaterloo ECE Work Term Report Template LaTeX source code
% Original Microsoft Word Document: <https://ece.uwaterloo.ca/~dwharder/Reports/Word/Report.Template.docx>
%
% Maintainer: Xin (Golson) Xie <golson.xie@uwaterloo.ca>
% Last Modified: 2014-06-28
% URL: <https://www.sharelatex.com/project/51b785e04c2bd70430657845>
% Documentation: <https://docs.google.com/document/d/1V9h24vhnaTfnRmiQWF3NaVye0RnbkpDhSldMcDREbNw>

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Identification                                                             %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ece}[2014/02/22 UWaterloo ECE WRRPT Template Class]

\LoadClass{report}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Required packages                                                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.5" left margin, 1" right,top,bottom margins
\RequirePackage[top=1in,bottom=1in,left=1.5in,right=1in]{geometry}

% web address and email, as well as the content list, i.e. structure in PDF
\RequirePackage{hyperref}

% set spacing
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{parskip}

% modify cftfigfont and cfttabfont after body is load (so that the modification
% to ToC, LoF, and LoT below will take effect)
\RequirePackage[titles]{tocloft}

% set numbers of ToC
\RequirePackage{bookmark}

% modify format of caption
\RequirePackage{caption}

% micro-typographic extension
\RequirePackage{microtype}

% set date format to "12 June 2013"
\RequirePackage[UKenglish]{isodate}

% enumerate and itemize
\RequirePackage{enumitem}

% default math packages
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}

% equation referencing
\RequirePackage[amsmath,thmmarks]{ntheorem}

% for underline fonts
\RequirePackage[normalem]{ulem}

% for using font closer to Times New Roman (instead of "genuine" Roman of LaTeX). Thanks to Sagar Rajendran for pointing out
\RequirePackage{mathptmx}

% for more clever way of referencing (must be loaded after hyperref package)
\RequirePackage[noabbrev]{cleveref}

% SI Units
\RequirePackage{siunitx}

% figures importing
\RequirePackage[pdftex]{graphicx}

% glossaries (must be loaded after hyperref package) and style
\RequirePackage[nonumberlist,section]{glossaries}
\RequirePackage{glossary-tree}

% reference, let all entries start with numbers
\RequirePackage[backend=bibtex,style=numeric]{biblatex}

% appendix
\RequirePackage{appendix}

% redine \section*
\RequirePackage{ifthen}
\RequirePackage{suffix}

% prepend/append additional command to original command definition
\RequirePackage{etoolbox}

% algorithm
\RequirePackage[]{algorithm2e}

% source code listing
\RequirePackage{listings}

% define source code color
\RequirePackage{color}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class definition                                                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set size of letter paper
\setlength\paperheight{11in}
\setlength\paperwidth{8.5in}

% section headings in 12- to 14-point default Roman font
\titleformat*{\section}{\LARGE\bfseries}
\titleformat*{\subsection}{\Large\bfseries}
\titleformat*{\subsubsection}{\large\bfseries}

% Table of Contents, List of Figures, and List of Table entries are properly tab
% filled
\setlength{\cftsecindent}{0em}
\setlength{\cftsubsecindent}{1.25em}
\setlength{\cftsubsubsecindent}{2.5em}
\setlength{\cftfigindent}{0em}
\setlength{\cfttabindent}{0em}
\setlength{\cftaftertoctitleskip}{0.5\cftaftertoctitleskip}
\setlength{\cftafterloftitleskip}{0.5\cftafterloftitleskip}
\setlength{\cftafterlottitleskip}{0.5\cftafterlottitleskip}

% set space between paragraphs to be one line
\setlength{\parskip}{\baselineskip}

% make sure the subsubs are numbered and on the ToC
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

% reset the format of caption (add '.' after Figure X.X)
\captionsetup[figure]{labelsep=period}
\captionsetup[table]{labelsep=period}

% set date style
\cleanlookdateon

% no separation between items in itemize and enumerate
\setlist{noitemsep}

% two spaces after each sentence
\frenchspacing

% disable ligatures to avoid searching problem,
% e.g. "efficiency" will not be search-able PDF if it is not done so
\DisableLigatures{encoding = *, family = *}

% settings of source code
\definecolor{mygreen}{rgb}{0,0.6,0}
\definecolor{mygray}{rgb}{0.5,0.5,0.5}
\definecolor{mymauve}{rgb}{0.58,0,0.82}
\lstset{ %
    % choose the background color
    backgroundcolor=\color{white},
    % the size of the fonts that are used for the code
    basicstyle=\footnotesize\ttfamily,
    % sets if automatic breaks should only happen at whitespace
    breakatwhitespace=false,
    % sets automatic line breaking
    breaklines=true,
    % sets the caption-position to bottom
    captionpos=b,
    % comment style
    commentstyle=\color{mygreen},
    % if you want to add LaTeX within your code
    escapeinside={\%*}{*)},
    % lets you use non-ASCII characters; for 8-bits encodings only, does not work with UTF-8
    extendedchars=true,
    % adds a frame around the code
    frame=single,
    % keeps spaces in text, useful for keeping indentation of code (possibly needs columns=flexible)
    keepspaces=true,
    % keyword style
    keywordstyle=\color{blue},
    % where to put the line-numbers; possible values are (none, left, right)
    numbers=left,
    % how far the line-numbers are from the code
    numbersep=5pt,
    % the style that is used for the line-numbers
    numberstyle=\tiny\color{mygray},
    % if not set, the frame-color may be changed on line-breaks within not-black text (e.g. comments (green here))
    rulecolor=\color{black},
    % show spaces everywhere adding particular underscores; it overrides 'showstringspaces'
    showspaces=false,
    % underline spaces within strings only
    showstringspaces=false,
    % show tabs within strings adding particular underscores
    showtabs=false,
    % the step between two line-numbers. If it's 1, each line will be numbered
    stepnumber=1,
    % string literal style
    stringstyle=\color{mymauve},
    % sets default tabsize to 2 spaces
    tabsize=2,
    % show the filename of files included with \lstinputlisting; also try caption instead of title
    title=\lstname
}

% set reference format for appendices
\crefformat{appendix}{#2#1#3}
\Crefformat{appendix}{#2#1#3}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Original command re-definitions                                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set Roman as default font
\renewcommand{\familydefault}{\rmdefault}

% use "1" as section number instead of "0.1"
% report body and appendices use section and subsection numbering
\renewcommand{\thesection}{\arabic{section}}

% change the numbering of figures and tables (Figure 1 --> Figure 1-1)
\renewcommand{\thefigure}{\thesection-\arabic{figure}}
\renewcommand{\thetable}{\thesection-\arabic{table}}

% format List of Figures and List of Tables as "Figure X.X" and "Table X.X"
% report body use section and subsection numbering
\renewcommand{\cftfigfont}{Figure }
\renewcommand{\cfttabfont}{Table }

% For ToC, LoF, LoT title font
\renewcommand{\cfttoctitlefont}{\Large\textbf}
\renewcommand{\cftloftitlefont}{\Large\textbf}
\renewcommand{\cftlottitlefont}{\Large\textbf}
% If want title of ToC, LoF, LoT centred, comment previous 3 lines and uncomment
% below 3 lines
%\renewcommand{\cfttoctitlefont}{\hfil\Large\textbf}
%\renewcommand{\cftloftitlefont}{\hfil\Large\textbf}
%\renewcommand{\cftlottitlefont}{\hfil\Large\textbf}

\renewcommand{\contentsname}{Table of Contents}

\renewcommand{\maketitle}{
    % front matter, except for Contributions and Summary, single line spacing
    \singlespacing
        
    % front matter page numbers lower case roman
    % set the page numbers to be roman
    \pagenumbering{roman}
        
    % set no page number in title page
    \thispagestyle{empty}
    \begin{center}
        \huge{Technische Universiteit Delft}\\
        \LARGE{Faculty of Electrical Engineering, Mathematics and Computer Science}\\
        \normalsize % without this line, the baselineskip next line will be of Large font
        \vspace*{8\baselineskip}
        \HUGE{\printreporttitle}\\
        \normalsize
        \vspace*{\baselineskip}
        \normalsize{\printselfstudy}\\
        \vspace*{8\baselineskip}
        \Large{\printemployername}\\
        \normalsize{\printemployerstreetaddress}\\
        \normalsize{\printemployerlocation}\\
        \vspace*{12\baselineskip}
        \normalsize{Prepared by}\\
        \normalsize{\printauthorname}\\
        \normalsize{\printstudentnumber}\\
        \normalsize{\printuserid}\\
        \normalsize{\printterm\:\printprogram}\\
        \normalsize{\today}\\
        \vspace*{\baselineskip}
        \normalsize{\printconfidential}\\
    \end{center}
}

% Adjust gap above lists
\preto\enumerate{\vspace{0.5\baselineskip}}
\preto\itemize{\vspace{0.5\baselineskip}}

% redefine and add more formatting on ToC, LoF, and LoC
\preto\tableofcontents{
    % Table of Contents on a separate page
    \clearpage
    % ToC single spacing
    \singlespacing
}
\preto\listoffigures{
    % List of Figures on a separate page
    \clearpage
    % add LoF to ToC
    \csname phantomsection\endcsname
    \addcontentsline{toc}{section}{\listfigurename}
    % LoF single spacing
    \singlespacing
}
\preto\listoftables{
    % List of Tables on a separate page
    \clearpage
    % add LoT to ToC
    \csname phantomsection\endcsname
    \addcontentsline{toc}{section}{\listtablename}
    % LoT single spacing
    \singlespacing
}

% adjust section number and content format for appendices
\appto\appendix{
    \addtocontents{toc}{\string\renewcommand\string\l@section{\string\@dottedtocline{2}{0em}{6em}}}
    \renewcommand{\thesection}{Appendix \Alph{section}}
    \preto\section{\clearpage}
}

% renew the font size, the number in the 2 brackets are font size and skip size
% catch: skip size = 1.2 x font size
\renewcommand{\normalsize}{\@setfontsize\normalsize{11pt}{13.2}}
\renewcommand{\large}{\@setfontsize\large{12pt}{14.4}}
\renewcommand{\Large}{\@setfontsize\Large{13pt}{15.6}}
\renewcommand{\LARGE}{\@setfontsize\LARGE{14pt}{16.8}}
\renewcommand{\huge}{\@setfontsize\huge{16pt}{19.2}}
\renewcommand{\Huge}{\@setfontsize\Huge{18pt}{21.6}}
\newcommand{\HUGE}{\@setfontsize\HUGE{20pt}{24}}

% reset the figure and table number at each section (or after Table 1-2, the
% table in Section 2 becomes Table 2-3 without resetting)
\@addtoreset{figure}{section}
\@addtoreset{table}{section}

% redefine \section* without changing \section
\let\old@section\section
\WithSuffix\def\section*{
    \ifx\next
    \let\next\report@section@star
    \else
    \def\next{\report@section@star[]}
    \fi
    \next
}
\def\report@section@star[#1]#2{
    \clearpage
    \ifthenelse
    {\equal{#1}{}}
    {\def\report@section@short{#2}\old@section*{#2}}
    {\def\report@section@short{#1}\old@section*[#1]{#2}}
    \addcontentsline{toc}{section}{\report@section@short}
}

% adjust glossary formatting
\preto\printglossaries{
    \singlespacing
}

% adjust reference formatting
\preto\printbibliography{
    \references
    \section*{References}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Command definitions                                                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% modified the Glossary style a bit, i.e. add colon after entry name and add
% space for entries in same group (by initial letter)
\newglossarystyle{ieeestyle}{
    \glossarystyle{tree}
    \renewcommand{\glsnamefont}[1]{\textbf{##1:}}
    \renewenvironment{theglossary}{\setlength{\parskip}{\baselineskip}}{}
    \renewcommand{\glsgroupskip}{}
}

% define some "variables"
\newcommand{\printreporttitle}{\quad}
\newcommand{\printselfstudy}{\quad}
\newcommand{\printemployername}{\quad}
\newcommand{\printemployerstreetaddress}{\quad}
\newcommand{\printemployerlocation}{\quad}
\newcommand{\printauthorname}{\quad}
\newcommand{\printstudentnumber}{\quad}
\newcommand{\printuserid}{\quad}
\newcommand{\printterm}{\quad}
\newcommand{\printprogram}{\quad}
\newcommand{\printconfidential}{\quad}
\newcommand{\printauthorstreetaddress}{\quad}
\newcommand{\printauthorlocation}{\quad}
\newcommand{\printauthorpostalcode}{\quad}
\newcommand{\reporttitle}[1]{\renewcommand{\printreporttitle}{#1}}
\newcommand{\selfstudy}{\renewcommand{\printselfstudy}{Self Study}}
\newcommand{\employername}[1]{\renewcommand{\printemployername}{#1}}
\newcommand{\employerstreetaddress}[1]{\renewcommand{\printemployerstreetaddress}{#1}}
\newcommand{\employerlocation}[1]{\renewcommand{\printemployerlocation}{#1}}
\newcommand{\authorname}[1]{\renewcommand{\printauthorname}{#1}}
\newcommand{\studentnumber}[1]{\renewcommand{\printstudentnumber}{Student ID #1}}
\newcommand{\userid}[1]{\renewcommand{\printuserid}{Kaggle ID #1}}
\newcommand{\term}[1]{\renewcommand{\printterm}{#1\:}}
\newcommand{\program}[1]{\renewcommand{\printprogram}{#1}}
\newcommand{\confidential}[1]{\renewcommand{\printconfidential}{Confidential-#1}}
\newcommand{\authorstreetaddress}[1]{\renewcommand{\printauthorstreetaddress}{#1}}
\newcommand{\authorlocation}[1]{\renewcommand{\printauthorlocation}{#1}}
\newcommand{\authorpostalcode}[1]{\renewcommand{\printauthorpostalcode}{#1}}

% define some section formatting templates
\newcommand{\letterofsubmittal}{
    \clearpage
    % normal size font from here on
    \normalsize
    % no page number in cover letter
    \thispagestyle{empty}
    % Cover Letter single spacing
    \singlespacing
}
\newcommand{\contributions}{
    % Contributions 1.5 spacing
    \onehalfspacing
    \section*{Contributions}
}
\newcommand{\summary}{
    % Summary 1.5 spacing
    \onehalfspacing
    \section*{Summary}
}
\newcommand{\body}{
    \clearpage
    % restart page numbering at start of report body
    % set the page numbers to be arabic, starting at page 1 %
    \setcounter{page}{1}
    \pagenumbering{arabic}
    % report body 1.5 spacing
    \onehalfspacing
}
\newcommand{\references}{
    % References single spacing
    \singlespacing
}

% little helper command used to check font size
%\newcommand{\thefontsize}{The current font size is: \f@size pt}

\glossarystyle{ieeestyle}
\makeglossaries

%% end of file `ece.cls'.

