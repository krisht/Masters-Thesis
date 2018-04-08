%-------------------------- identification ---------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ociamthesis}[2010/11/22 v2.2 OCIAM thesis class]
\newcommand*{\SignatureAndDate}[1]{%
    \par\noindent\makebox[3in]{\hrulefill}
    \par\noindent\makebox{#1}
}%% default square logo 
%-------------------------- initial code -----------------------
\def\logoversion{squarelogo}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[a4paper]{report}

\RequirePackage{graphicx} % needed for latest frontpage logo

\raggedbottom

%define the default submitted text
\newcommand{\submittedtext}{{A thesis submitted for the degree of}}

\newcommand\textline[3][t]{%
  \par\smallskip\noindent\parbox[#1]{.5\textwidth}{\center#2}
  \par
  \parbox[#1]{.5\textwidth}{\center#3}\par
}

\renewcommand{\chaptermark}[1]{\markboth{#1}{}}

% The year and term the thesis is submitted 
\def\degreedate#1{\gdef\@degreedate{#1}}
% The full (unabbreviated) name of the degree
\def\degree#1{\gdef\@degree{#1}}
% The name of your Oxford college (e.g. Christ Church, Pembroke)
\def\college#1{\gdef\@college{#1}}



\def\logo{{\includegraphics[width=32mm]{cooperlogo}}}

\setlength{\topmargin}{0.0in}
\setlength{\oddsidemargin}{0.33in}
\setlength{\evensidemargin}{-0.08in}
\setlength{\textheight}{9.0in}
\setlength{\textwidth}{6.0in}

\newenvironment{alwayssingle}{%
       \@restonecolfalse
       \if@twocolumn\@restonecoltrue\onecolumn
       \else\if@openright\cleardoublepage\else\clearpage\fi
       \fi}%
       {\if@restonecol\twocolumn
       \else\newpage\thispagestyle{empty}\fi}

%define title page layout
\renewcommand{\maketitle}{%
\begin{alwayssingle}
    \renewcommand{\footnotesize}{\small}
    \renewcommand{\footnoterule}{\relax}
    \thispagestyle{empty}

    \begin{center}
      {\Large \textsc{THE COOPER UNION \\ ALBERT NERKEN SCHOOL OF ENGINEERING} \par}
      \vspace*{5em}
      { \huge \textsc{\@title} \par}

      \vspace*{5em}
      \textline[t]{\fontsize{19}{24} \selectfont by}
      {\fontsize{19}{24} \selectfont Radhakrishnan Thiyagarajan}
      \vspace*{5em}
      
      {\fontsize{19}{24} \selectfont {\submittedtext} \par}
      {\fontsize{19}{24} \selectfont \it {\@degree} \par}

      \vspace*{5em}
      {\fontsize{19}{24} \selectfont \@degreedate}

      \vspace*{5em}
      {\fontsize{19}{24} \selectfont Professor Sam Keene, Advisor}

    \end{center}



%     \renewcommand{\footnotesize}{\small}
%     \renewcommand{\footnoterule}{\relax}
%     \thispagestyle{empty}
%   \begin{center}
%   {\LARGE \textsc{masters thesis}\par \vspace*{5mm}}
%     { \huge {\bfseries {\@title}} \par}
% {\large \vspace*{20mm} {\logo}}

%     \textline[t]{\large \textit{Author}}{\large Radhakrishnan Thiyagarajan}
%     \textline[t]{\large \textit{Adviser}}{\large Sam Keene}

% {\large \vspace*{20mm}
%     {{\@college} \par}
% \vspace*{1ex}
%     {The Cooper Union for the Advancement of Science \& Art \par}
% \vspace*{10mm}
%     {{\submittedtext} \par}
% \vspace*{1ex}
%     {\it {\@degree} \par}
% \vspace*{2ex}
%     {\@degreedate}}
%   \end{center}


\end{alwayssingle}

}

% APPROVAL

\newenvironment{approval}
{\begin{alwayssingle}
  \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\LARGE }
  \end{center}
  \vspace{0.5cm}
  \begin{verse}\begin{center}}
{\end{center}\end{verse}\end{alwayssingle}}

% DEDICATION
\newenvironment{dedication}
{\begin{alwayssingle}
  \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\LARGE }
  \end{center}
  \vspace{0.5cm}
  \begin{verse}\begin{center}}
{\end{center}\end{verse}\end{alwayssingle}}


% ACKNOWLEDGEMENTS

\newenvironment{acknowledgements}
{\begin{alwayssingle} \thispagestyle{empty}
\begin{center}
\vspace*{1.5cm}
{\Large \bfseries Acknowledgements}
\end{center}
\vspace{0.5cm}
\begin{quote}}
{\end{quote}\end{alwayssingle}}

% The acknowledgementslong environment puts a large, bold, centered 
% "Acknowledgements" label at the top of the page. The acknowledgement itself 
% does not appears in a quote environment so you can get more in.

\newenvironment{acknowledgementslong}
{\begin{alwayssingle} \thispagestyle{empty}
\begin{center}
\vspace*{1.5cm}
{\Large \bfseries Acknowledgements}
\end{center}
\vspace{0.5cm}}
{\end{alwayssingle}}

%ABSTRACT


\renewenvironment{abstract} {\begin{alwayssingle} \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\Large \bfseries  Abstract}
  \end{center}
  \vspace{0.5cm}
   \begin{quote}}
{\end{quote}\end{alwayssingle}}

\newenvironment{abstractlong} {\begin{alwayssingle} \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\Large \bfseries  Abstract}
  \end{center}
  \vspace{0.5cm}}
{\end{alwayssingle}}

\newenvironment{abstractseparate} {\begin{alwayssingle} \thispagestyle{empty}
  \vspace*{-1in}
 \begin{center}
 

 {\Large \bfseries  Abstract \\ \par}

 
    { \Large {\bfseries {\@title}} \par}
    {{\large \vspace*{1ex} \@author} \par}
    {\it {\@degree} \par}
    
    
  \end{center}}
{\end{alwayssingle}}

\newenvironment{romanpages}
{\cleardoublepage\setcounter{page}{1}\renewcommand{\thepage}{\roman{page}}}
{\cleardoublepage\renewcommand{\thepage}{\arabic{page}}\setcounter{page}{1}}

\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain}{}{\sl{\rightmark}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}