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

\def\changemargin#1#2{\list{}{\rightmargin#2\leftmargin#1}\item[]}
\let\endchangemargin=\endlist

\newcommand\textline[3][t]{%
\par\smallskip\noindent\parbox[#1]{.5\textwidth}{\center#2}
\par
\parbox[#1]{.5\textwidth}{\center#3}\par
}

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
{\large \textsc{THE COOPER UNION \\ ALBERT NERKEN SCHOOL OF ENGINEERING} \par}
\vspace*{5em}
{ \Huge \textbf{\textsc{\@title}} \par}

\vspace*{0.1in}
\textline[t]{\fontsize{19}{24} \selectfont by}
{\fontsize{19}{24} \selectfont Radhakrishnan Thiyagarajan}
\vspace*{5em}

\vspace*{3em}
{\fontsize{19}{24} \selectfont \@degreedate}
\vspace*{7em}

{\fontsize{19}{24} \selectfont {\submittedtext} \par}
{\fontsize{19}{24} \selectfont \it {\@degree} \par}



\vspace*{2em}
{\fontsize{19}{24} \selectfont Advisor \\ \vspace{0.125in} Professor Sam Keene}

\end{center}



\end{alwayssingle}

}


\newenvironment{romanpages}
{\cleardoublepage\setcounter{page}{1}\renewcommand{\thepage}{\roman{page}}}
{\cleardoublepage\renewcommand{\thepage}{\arabic{page}}\setcounter{page}{1}}
