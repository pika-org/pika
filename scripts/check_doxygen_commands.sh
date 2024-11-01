#!/usr/bin/env bash

# Copyright (c) 2024 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -u

# All commands from https://www.doxygen.nl/manual/commands.html. Last updated 2024-11-01. Note that
# some commands must be escaped.
rg --type=cpp --type=cuda '^\s*(///|\*).*@(a|addindex|addtogroup|anchor|arg|attention|author|authors|b|brief|bug|c|callergraph|callgraph|category|cite|class|code|collaborationgraph|concept|cond|copybrief|copydetails|copydoc|copyright|date|def|defgroup|deprecated|details|diafile|dir|directorygraph|docbookinclude|docbookonly|dontinclude|dot|dotfile|doxyconfig|e|else|elseif|em|emoji|endcode|endcond|enddocbookonly|enddot|endhtmlonly|endif|endinternal|endlatexonly|endlink|endmanonly|endmsc|endparblock|endrtfonly|endsecreflist|endverbatim|enduml|endxmlonly|enum|example|exception|extends|f\(|f\)|f\$|f\[|f\]|f\{|f\}|file|fileinfo|fn|groupgraph|headerfile|hidecallergraph|hidecallgraph|hidecollaborationgraph|hidedirectorygraph|hideenumvalues|hidegroupgraph|hideincludedbygraph|hideincludegraph|hideinheritancegraph|hideinlinesource|hiderefby|hiderefs|hideinitializer|htmlinclude|htmlonly|idlexcept|if|ifnot|image|implements|important|include|includedoc|includedbygraph|includegraph|includelineno|ingroup|inheritancegraph|internal|invariant|interface|latexinclude|latexonly|li|line|lineinfo|link|mainpage|maninclude|manonly|memberof|module|msc|mscfile|n|name|namespace|noop|nosubgrouping|note|overload|p|package|page|par|paragraph|param|parblock|post|pre|private|privatesection|property|protected|protectedsection|protocol|public|publicsection|pure|qualifier|raisewarning|ref|refitem|related|relates|relatedalso|relatesalso|remark|remarks|result|return|returns|retval|rtfinclude|rtfonly|sa|secreflist|section|see|short|showdate|showenumvalues|showinitializer|showinlinesource|showrefby|showrefs|since|skip|skipline|snippet|snippetdoc|snippetlineno|static|startuml|struct|subpage|subparagraph|subsection|subsubparagraph|subsubsection|tableofcontents|test|throw|throws|todo|tparam|typedef|plantumlfile|union|until|var|verbatim|verbinclude|version|vhdlflow|warning|weakgroup|xmlinclude|xmlonly|xrefitem|\$|@|\\|&|~|<|=|>|#|%|"|\.|::|\||--|---)\b' "${1}"

if [[ $? != 1 ]]; then
    echo "Use \command instead of @command in doxygen comments"
    exit 1
fi
