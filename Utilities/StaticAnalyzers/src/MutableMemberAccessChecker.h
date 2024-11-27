//== MutableMemberChecker.h - Checks for mutable members --------------*- C++ -*--==//
//
// by Thomas Hauth [ Thomas.Hauth@cern.ch ]
//
//===----------------------------------------------------------------------===//

#ifndef Utilities_StaticAnalyzers_MutableMemberChecker_h
#define Utilities_StaticAnalyzers_MutableMemberChecker_h

#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CmsException.h"
#include "CmsSupport.h"

namespace clangcms {
  class MutableMemberChecker : public clang::ento::Checker<clang::ento::check::PreStmt<
                                   clang::MemberExpr /*,
                                                           clang::ento::check::ASTDecl<clang::FieldDecl>,
                                                           clang::ento::check::EndAnalysis */>> {
    // private:
    //   mutable llvm::DenseSet<const clang::FieldDecl *> MutableMembers;
    //   mutable llvm::DenseSet<const clang::FieldDecl *> ModifiedMutableMembers;

  public:
    CMS_SA_ALLOW mutable std::unique_ptr<clang::ento::BugType> BT;
    void checkPreStmt(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const;
    // void checkASTDecl(const clang::FieldDecl *D, clang::ento::AnalysisManager &Mgr, clang::ento::BugReporter &BR) const;
    // void checkEndAnalysis(clang::ento::ExplodedGraph &G,
    //                       clang::ento::BugReporter &BR,
    //                       clang::ento::ExprEngine &Eng) const;

  private:
    CmsException m_exception;
    bool checkAssignToMutable(const clang::MemberExpr *ME,
                              clang::ento::CheckerContext &C,
                              const clang::FunctionDecl *FuncD) const;
    bool checkCallNonConstOfMutable(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const;
    // void reportUselessMutableField(const clang::FieldDecl *Field, clang::ento::BugReporter &BR) const;
  };
}  // namespace clangcms

#endif
