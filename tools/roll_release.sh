#!/usr/bin/env bash
#
# Copyright (c) 2019-2022 ETH Zurich
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

# This script tags a release locally and creates a release on GitHub. It relies
# on the hub command line tool (https://hub.github.com/).

set -o errexit

VERSION_MAJOR=$(sed -n 's/set(PIKA_VERSION_MAJOR \(.*\))/\1/p' CMakeLists.txt)
VERSION_MINOR=$(sed -n 's/set(PIKA_VERSION_MINOR \(.*\))/\1/p' CMakeLists.txt)
VERSION_PATCH=$(sed -n 's/set(PIKA_VERSION_PATCH \(.*\))/\1/p' CMakeLists.txt)
VERSION_TAG=$(sed -n 's/set(PIKA_VERSION_TAG "\(.*\)")/\1/p' CMakeLists.txt)
VERSION_FULL_NOTAG=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
VERSION_FULL_TAG=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_TAG}
VERSION_TITLE="pika ${VERSION_FULL_NOTAG}"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if ! which hub >/dev/null 2>&1; then
    echo "Hub not installed on this system (see https://hub.github.com/). Exiting."
    exit 1
fi

if ! [[ "$CURRENT_BRANCH" =~ ^release-[0-9]+\.[0-9]+\.X$ ]]; then
    echo "Not on release branch (current branch is \"${CURRENT_BRANCH}\"). Not continuing to make release."
    exit 1
fi

printf "Checking that the git repository is in a clean state... "
if [[ $(git status --porcelain | wc -l) -eq 0 ]]; then
    echo "OK"
else
    echo "ERROR"
    git status -s
    echo "Do you want to continue anyway?"
    select yn in "Yes" "No"; do
        case $yn in
        Yes) break ;;
        No) exit ;;
        esac
    done
fi

changelog_path="docs/changelog.md"

if [ -z "${VERSION_TAG}" ]; then
    echo "You are about to tag and create a final release on GitHub."

    echo ""
    echo "Sanity checking release"

    sanity_errors=0

    printf "Checking that %s has an entry for %s... " "${changelog_path}" "${VERSION_FULL_NOTAG}"
    if grep "## ${VERSION_FULL_NOTAG}" "${changelog_path}"; then
        echo "OK"
    else
        echo "Missing"
        sanity_errors=$((sanity_errors + 1))
    fi

    printf "Checking that %s also has a date set for %s... " "${changelog_path}" "${VERSION_FULL_NOTAG}"
    if grep "## ${VERSION_FULL_NOTAG} ([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\})" "${changelog_path}"; then
        echo "OK"
    else
        echo "Missing"
        sanity_errors=$((sanity_errors + 1))
    fi

    if [[ ${sanity_errors} -gt 0 ]]; then
        echo "Found ${sanity_errors} error(s). Fix it/them and try again."
        exit 1
    fi
else
    echo "You are about to tag and create a pre-release on GitHub."
    echo "If you intended to make a final release, remove the tag in the main CMakeLists.txt first."
fi

# Extract the changelog for this version
VERSION_DESCRIPTION=$(
    # Find the correct heading and print everything from there to the end of the file
    awk "/^## ${VERSION_FULL_NOTAG}/,EOF" ${changelog_path} |
        # Remove the heading
        tail -n+3 |
        # Find the next heading or the end of the file and print everything until that heading
        sed '/^## /Q'
)

echo ""
echo "The version is: ${VERSION_FULL_TAG}"
echo "The version title is: ${VERSION_TITLE}"
echo "The version description is:"
echo "${VERSION_DESCRIPTION}"

echo "Do you want to continue?"
select yn in "Yes" "No"; do
    case $yn in
    Yes) break ;;
    No) exit ;;
    esac
done

if [[ -z "${GITHUB_USER}" || -z "${GITHUB_PASSWORD}" ]] && [[ -z "${GITHUB_TOKEN}" ]]; then
    echo "Need GITHUB_USER and GITHUB_PASSWORD or only GITHUB_TOKEN to be set to use hub release."
    exit 1
fi

if [ -z "${VERSION_TAG}" ]; then
    PRERELEASE_FLAG=""
else
    PRERELEASE_FLAG="--prerelease"
fi

echo ""
if [[ "$(git tag -l ${VERSION_FULL_TAG})" == "${VERSION_FULL_TAG}" ]]; then
    echo "Tag already exists locally."
else
    echo "Tagging release."
    git tag --annotate "${VERSION_FULL_TAG}" --message="${VERSION_TITLE}"
fi

remote=$(git remote -v | grep pika-org\/pika.git | cut -f1 | uniq)
if [[ "$(git ls-remote --tags --refs $remote | grep -o ${VERSION_FULL_TAG}.*)" == "${VERSION_FULL_TAG}" ]]; then
    echo "Tag already exists remotely."
else
    echo "Pushing tag to $remote."
    git push $remote "${VERSION_FULL_TAG}"
fi

echo ""
echo "Creating release."
hub release create \
    ${PRERELEASE_FLAG} \
    --message="${VERSION_TITLE}" \
    --message="${VERSION_DESCRIPTION}" \
    "${VERSION_FULL_TAG}"
