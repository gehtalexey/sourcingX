-- Migration 031: public.sourcingx_upsert_profiles - the Crustdata-id dedupe
-- also reads the free ids in public.supanova_person_ids.
--
-- STATUS: NOT YET APPLIED. Written 2026-11-18. Needs a real Codex review and
-- Alexey's go-ahead before it touches the shared Supabase database.
-- Apply migrations/manual/031_supanova_person_ids_person_id_index.sql first.
--
-- WHY
-- Since 027/028 the save looks for an existing row with the same Crustdata id,
-- but only inside public.profiles.raw_data. 18,850 Israeli rows keep their id
-- only in Supanova's table public.supanova_person_ids (recovered free on
-- 2026-11-17; where both exist they agree 22,143 / 22,143). The check misses
-- those rows, so ~15 second copies were saved since 2026-11-03.
-- Plan: supanova docs/investigations/plan-save-check-reads-free-ids-2026-11-18.md
--
-- WHAT CHANGED vs the live function (taken from pg_get_functiondef on
-- 2026-11-18; everything else is identical, character for character)
-- - After the existing raw_data id lookup finds no row (v_redir still false),
--   a second lookup reads public.supanova_person_ids for that person_id and
--   joins to public.profiles on linkedin_url. Rules:
--     * method must be set and must NOT be 'fuzzy_israel' or
--       'unmatched_2026_11_18:name_strict_check' (the 85 guessier matches;
--       a wrong id would write a new person into someone else's row);
--     * the target row must have NO Crustdata id of its own in raw_data
--       (if it had one equal to ours, the first lookup would have found it;
--       if it had a different one, the two sources disagree, so skip it);
--     * the target's LinkedIn handle must differ from the incoming one
--       (same rule as the first lookup);
--     * same tie-break order as the first lookup.
-- - A hit redirects the save exactly like the first lookup: same v_ident /
--   v_redir handling, so the same alias merge of the sent URL runs. The
--   advisory lock on the Crustdata id is the same one already taken for the
--   whole batch. The lookup sits inside the same exception block, so a failure
--   falls back to the sent URL, as today.
-- - Signature, results, grants and ordering unchanged.
--
-- SHARED DATABASE: agent-kalamata's public.israel_autopilot_upsert_profiles
-- needs the same change in its own PR.
--
-- Rollback: re-run CREATE OR REPLACE with the body from
-- migrations/028_sourcingx_upsert_profiles_nested_crustdata_id.sql (same
-- signature). The index can stay.

-- Prerequisite check. public.supanova_person_ids is Supanova's table and no
-- migration in this repo creates it. PL/pgSQL plans the lookup only at run
-- time, and the lookup sits inside an exception block, so without this check a
-- database that lacks the table would accept the function and then silently
-- skip the new dedupe on every save. Fail here instead, loudly.
DO $check$
BEGIN
  IF to_regclass('public.supanova_person_ids') IS NULL THEN
    RAISE EXCEPTION 'migration 031: public.supanova_person_ids does not exist (Supanova table); the new save check needs it';
  END IF;
  IF to_regclass('public.idx_supanova_person_ids_person_id') IS NULL THEN
    RAISE EXCEPTION 'migration 031: apply migrations/manual/031_supanova_person_ids_person_id_index.sql first';
  END IF;
END
$check$;

CREATE OR REPLACE FUNCTION public.sourcingx_upsert_profiles(p_profiles jsonb)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
 SET statement_timeout TO '20s'
AS $function$
declare
  v_max_batch constant int  := 100;
  v_writer    constant text := 'sourcingx';
  v_source    constant text := 'crustdata-enrich';
  v_provider  constant text := 'crustdata';
  v_schema_default constant text := 'crustdata-2025-11-01';
  v_total    int;
  v_run      uuid := pg_catalog.gen_random_uuid();
  v_approval text := 'SOURCINGX-' || to_char(now(), 'YYYYMMDD');
  v_elem     jsonb;
  v_prof     jsonb;
  v_sent     text;
  v_pid      uuid;
  v_action   text;
  v_curl     text;
  v_err      text;
  v_aliases  text[];
  v_alias_n  int;
  v_lock_cid text;
  v_cid      text;
  v_target   text;
  v_ident    text;
  v_redir    boolean;
  v_redir_n  int := 0;
  v_results  jsonb := '[]'::jsonb;
  v_ins int := 0; v_upd int := 0; v_noop int := 0; v_rej int := 0;
  v_err_n int := 0; v_unk int := 0;
begin
  if p_profiles is null or jsonb_typeof(p_profiles) <> 'array' then
    raise exception 'sourcingx_upsert_profiles: p_profiles must be a jsonb array (got %)',
      coalesce(jsonb_typeof(p_profiles), 'null');
  end if;
  v_total := jsonb_array_length(p_profiles);
  if v_total = 0 then
    return jsonb_build_object('received', 0, 'written', 0, 'results', '[]'::jsonb);
  end if;
  if v_total > v_max_batch then
    raise exception 'sourcingx_upsert_profiles: batch of % profiles exceeds the % cap per call',
      v_total, v_max_batch;
  end if;
  insert into sourcing_raw.profile_ingestion_runs
         (id, approval_id, run_name, writer, source, provider, schema_version,
          status, input_count, started_at, notes)
  values (v_run, v_approval, v_writer || '-upsert-' || v_run::text,
          v_writer, v_source, v_provider, v_schema_default,
          'started', v_total, now(),
          'via public.sourcingx_upsert_profiles');
  for v_lock_cid in
    select distinct btrim(coalesce(
             nullif(btrim(e.elem -> 'raw_data' ->> 'person_id'), ''),
             nullif(btrim(e.elem -> 'raw_data' ->> 'crustdata_person_id'), ''),
             nullif(btrim(e.elem -> 'raw_data' -> '_new_api' ->> 'crustdata_person_id'), ''))) as cid
      from jsonb_array_elements(p_profiles) as e(elem)
     where jsonb_typeof(e.elem) = 'object'
     order by 1
  loop
    if v_lock_cid ~ '^[0-9]+$' and v_lock_cid !~ '^0+$' then
      perform pg_advisory_xact_lock(hashtextextended('sourcing-profiles-crustdata-id:' || v_lock_cid, 0));
    end if;
  end loop;
  for v_elem in
    select e.elem
      from jsonb_array_elements(p_profiles) with ordinality as e(elem, ord)
     order by e.ord
  loop
    v_pid := null; v_action := null; v_curl := null; v_err := null; v_sent := null;
    v_aliases := null; v_alias_n := null;
    v_cid := null; v_target := null; v_ident := null; v_redir := false;
    if jsonb_typeof(v_elem) <> 'object' then
      v_action := 'rejected';
      v_err := 'local-validation: array element is not a JSON object';
    elsif coalesce(btrim(v_elem ->> 'linkedin_url'), '') = '' then
      v_action := 'rejected';
      v_err := 'local-validation: missing linkedin_url';
    else
      v_sent := v_elem ->> 'linkedin_url';
      v_prof := public.engine_strip_contact(v_elem);
      v_ident := v_sent;
      begin
        v_cid := btrim(coalesce(
                   nullif(btrim(v_elem -> 'raw_data' ->> 'person_id'), ''),
                   nullif(btrim(v_elem -> 'raw_data' ->> 'crustdata_person_id'), ''),
                   nullif(btrim(v_elem -> 'raw_data' -> '_new_api' ->> 'crustdata_person_id'), '')));
        if v_cid ~ '^[0-9]+$' and v_cid !~ '^0+$' then
          select p.linkedin_url
            into v_target
            from public.profiles p
           where coalesce(nullif(p.raw_data->>'person_id',''), nullif(p.raw_data->>'crustdata_person_id',''), nullif(p.raw_data->'_new_api'->>'crustdata_person_id','')) = v_cid
             and coalesce(nullif(p.raw_data->>'person_id',''), nullif(p.raw_data->>'crustdata_person_id',''), nullif(p.raw_data->'_new_api'->>'crustdata_person_id','')) IS NOT NULL
             and sourcing_core.linkedin_handle(p.linkedin_url)
                   is distinct from sourcing_core.linkedin_handle(v_sent)
           order by (exists (select 1 from sourcing_core.profile_identity pi
                              where pi.canonical_profile_id = p.id)) desc,
                    p.created_at asc nulls last,
                    p.id asc
           limit 1;
          if nullif(btrim(v_target), '') is not null then
            v_ident := v_target;
            v_redir := true;
          end if;
          -- 031: no row holds this id in raw_data; try Supanova's free ids.
          if not v_redir then
            v_target := null;
            select p.linkedin_url
              into v_target
              from public.supanova_person_ids s
              join public.profiles p on p.linkedin_url = s.linkedin_url
             where s.person_id = v_cid
               and s.method is not null
               and s.method not in ('fuzzy_israel', 'unmatched_2026_11_18:name_strict_check')
               and coalesce(nullif(p.raw_data->>'person_id',''), nullif(p.raw_data->>'crustdata_person_id',''), nullif(p.raw_data->'_new_api'->>'crustdata_person_id','')) IS NULL
               and sourcing_core.linkedin_handle(p.linkedin_url)
                     is distinct from sourcing_core.linkedin_handle(v_sent)
             order by (exists (select 1 from sourcing_core.profile_identity pi
                                where pi.canonical_profile_id = p.id)) desc,
                      p.created_at asc nulls last,
                      p.id asc
             limit 1;
            if nullif(btrim(v_target), '') is not null then
              v_ident := v_target;
              v_redir := true;
            end if;
          end if;
        end if;
      exception when others then
        v_err := 'crustdata-id-lookup-failed: ' || SQLERRM;
        v_ident := v_sent;
        v_redir := false;
      end;
      begin
        select t.profile_id, t.action, t.canonical_url
          into v_pid, v_action, v_curl
          from sourcing_core.upsert_profile(
                 v_run, v_ident, v_prof, v_writer, v_source, v_provider,
                 coalesce(nullif(btrim(v_prof ->> 'schema_version'), ''), v_schema_default),
                 now()) t;
      exception when others then
        v_err := coalesce(v_err || '; ', '') || SQLERRM;
        v_action := 'error';
      end;
      if v_pid is not null and v_action in ('inserted', 'updated', 'noop')
         and (jsonb_typeof(v_elem -> 'alias_urls') = 'array' or v_redir) then
        begin
          select array_agg(distinct s.a)
            into v_aliases
            from (
              select btrim(elem_val #>> '{}') as a
                from jsonb_array_elements(
                       case when jsonb_typeof(v_elem -> 'alias_urls') = 'array'
                            then v_elem -> 'alias_urls' else '[]'::jsonb end) as elem_val
               where jsonb_typeof(elem_val) = 'string'
                 and nullif(btrim(elem_val #>> '{}'), '') is not null
              union all
              select btrim(v_sent) where v_redir
            ) s;
          if v_aliases is not null and array_length(v_aliases, 1) > 0 then
            v_alias_n := array_length(v_aliases, 1);
            update public.profiles p
               set original_urls = (
                 select array_agg(distinct a)
                   from unnest(coalesce(p.original_urls, '{}'::text[]) || v_aliases) as t(a)
                  where nullif(btrim(a), '') is not null)
             where p.id = v_pid;
          else
            v_alias_n := 0;
          end if;
        exception when others then
          v_err := coalesce(v_err || '; ', '') || 'alias-merge-failed: ' || SQLERRM;
          v_alias_n := 0;
        end;
      end if;
    end if;
    if v_redir then v_redir_n := v_redir_n + 1; end if;
    case coalesce(v_action, 'rejected')
      when 'inserted' then v_ins  := v_ins  + 1;
      when 'updated'  then v_upd  := v_upd  + 1;
      when 'noop'     then v_noop := v_noop + 1;
      when 'rejected' then v_rej  := v_rej  + 1;
      when 'error'    then v_err_n := v_err_n + 1;
      else v_unk := v_unk + 1;
    end case;
    v_results := v_results || jsonb_build_object(
      'sent_url',       v_sent,
      'canonical_url',  v_curl,
      'profile_id',     v_pid,
      'action',         coalesce(v_action, 'rejected'),
      'error',          v_err,
      'aliases_merged', v_alias_n,
      'redirected_from_crustdata_id', v_redir
    );
  end loop;
  update sourcing_raw.profile_ingestion_runs r
     set status = 'completed', finished_at = now(),
         inserted_count = v_ins, updated_count = v_upd,
         noop_count = v_noop, rejected_count = v_rej,
         conflict_count = v_err_n
   where r.id = v_run;
  return jsonb_build_object(
    'received',       v_total,
    'written',        v_ins + v_upd,
    'noop',           v_noop,
    'rejected',       v_rej,
    'errors',         v_err_n,
    'unknown_action', v_unk,
    'redirected',     v_redir_n,
    'ingestion_run',  v_run,
    'results',        v_results
  );
end
$function$;
