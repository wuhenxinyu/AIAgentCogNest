# Copyright (c) Alibaba, Inc. and its affiliates.
import copy

def merge_single_response(parsed_response, accumulated_data, n=1):
    """Merge a single response chunk with accumulated data.

    Args:
        parsed_response: The response chunk to merge
        accumulated_data: Dictionary storing accumulated data for each choice
        n: Number of expected choices (default 1)

    Returns:
        bool or list: True if this response should be yielded normally,
                      False if filtered, or a list of responses for n>1 with
                      non-stop finish reasons
    """
    # Check if all choices have been sent (for n > 1 case)
    if n > 1 and accumulated_data:
        all_sent = all(data.get('all_choices_sent', False)
                       for data in accumulated_data.values()
                       if isinstance(data, dict) and 'all_choices_sent' in data)
        if all_sent:
            return False

    # Track usage for each choice index when n > 1
    # Each streaming packet contains usage info for one specific choice
    if (n > 1 and parsed_response.usage and 
            parsed_response.output and parsed_response.output.choices and
            len(parsed_response.output.choices) > 0):
        if 'usage_by_index' not in accumulated_data:
            accumulated_data['usage_by_index'] = {}

        # Get the choice index from the first (and typically only) choice in this packet
        try:
            first_choice = parsed_response.output.choices[0]
            choice_idx = first_choice.index if hasattr(
                first_choice, 'index') and 'index' in first_choice else 0

            # Store only output_tokens for this choice index
            if 'output_tokens' in parsed_response.usage:
                accumulated_data['usage_by_index'][choice_idx] = dict(
                    parsed_response.usage)
        except (KeyError, AttributeError, IndexError):
            pass

    # Handle output.text accumulation when choices is null
    if (parsed_response.output and
            hasattr(parsed_response.output, 'text') and
            (not parsed_response.output.choices or parsed_response.output.choices is None)):
        choice_idx = 0
        if choice_idx not in accumulated_data:
            accumulated_data[choice_idx] = {
                'content': '',
                'reasoning_content': '',
                'tool_calls': [],
                'logprobs': {'content': []},
                'finished': False,
                'finish_reason': None,
                'all_choices_sent': False,
                'role': None
            }
        # Accumulate text if not empty
        if parsed_response.output.text:
            accumulated_data[choice_idx]['content'] += parsed_response.output.text
        # Always set accumulated content back to response
        parsed_response.output.text = accumulated_data[choice_idx]['content']
        return True

    # Process each choice in the choices array
    if parsed_response.output and parsed_response.output.choices:
        choices = parsed_response.output.choices

        # Filter out empty choices array
        if not choices:
            return False

        for choice_enum_idx, choice in enumerate(choices):
            # Use choice.index if available, otherwise use enumerate index
            try:
                choice_idx = choice.index if hasattr(choice, 'index') and 'index' in choice else choice_enum_idx
            except (KeyError, AttributeError):
                choice_idx = choice_enum_idx

            # Initialize accumulated data for this choice if not exists
            if choice_idx not in accumulated_data:
                accumulated_data[choice_idx] = {
                    'content': '',
                    'reasoning_content': '',
                    'tool_calls': [],
                    'logprobs': {'content': []},
                    'finished': False,
                    'finish_reason': None,
                    'all_choices_sent': False,
                    'role': None
                }

            # Handle message field - create if null
            if not choice.message:
                # Create message object with accumulated data
                choice.message = {
                    'role': accumulated_data[choice_idx]['role'] if accumulated_data[choice_idx]['role'] else 'assistant',
                    'content': accumulated_data[choice_idx]['content']
                }
                if accumulated_data[choice_idx]['reasoning_content']:
                    choice.message['reasoning_content'] = accumulated_data[choice_idx]['reasoning_content']
                if accumulated_data[choice_idx]['tool_calls']:
                    choice.message['tool_calls'] = accumulated_data[choice_idx]['tool_calls']
            else:
                # Save role if present
                if hasattr(choice.message, 'role') and choice.message.role:
                    accumulated_data[choice_idx]['role'] = choice.message.role

                # Handle content accumulation
                if 'content' in choice.message:
                    current_content = choice.message.content
                    if current_content:
                        # Check if content is multimodal format
                        if isinstance(current_content, list):
                            # Handle multimodal content (array format)
                            # Initialize accumulated content as array if not already
                            if not isinstance(accumulated_data[choice_idx]['content'], list):
                                accumulated_data[choice_idx]['content'] = []

                            # Ensure accumulated content list has enough elements
                            while len(accumulated_data[choice_idx]['content']) < len(current_content):
                                accumulated_data[choice_idx]['content'].append({'text': ''})

                            # Merge each content element
                            for content_idx, content_item in enumerate(current_content):
                                if isinstance(content_item, dict) and 'text' in content_item:
                                    if content_item['text']:
                                        # Accumulate text content
                                        accumulated_data[choice_idx]['content'][content_idx]['text'] += content_item['text']
                            # Update the current response with accumulated content
                            for content_idx in range(len(accumulated_data[choice_idx]['content'])):
                                if content_idx < len(choice.message.content):
                                    choice.message.content[content_idx]['text'] = accumulated_data[choice_idx]['content'][content_idx]['text']
                        else:
                            # Handle regular content (string format)
                            # Initialize accumulated content as string
                            if isinstance(accumulated_data[choice_idx]['content'], list):
                                accumulated_data[choice_idx]['content'] = ''
                            # Accumulate content if not empty
                            accumulated_data[choice_idx]['content'] += current_content
                    # Always set accumulated content back to response
                    if not isinstance(accumulated_data[choice_idx]['content'], list):
                        choice.message.content = accumulated_data[choice_idx]['content']
                    else:
                        # For multimodal content, ensure message.content
                        # exists
                        if not isinstance(choice.message.content, list):
                            choice.message.content = accumulated_data[choice_idx]['content']

                # Handle reasoning_content accumulation
                if 'reasoning_content' in choice.message:
                    current_reasoning_content = choice.message.reasoning_content
                    if current_reasoning_content:
                        accumulated_data[choice_idx]['reasoning_content'] += current_reasoning_content
                # Always set the accumulated reasoning_content back if we
                # have any, even if current response doesn't have it
                if accumulated_data[choice_idx]['reasoning_content']:
                    choice.message.reasoning_content = accumulated_data[choice_idx]['reasoning_content']

                # Handle tool_calls accumulation
                if 'tool_calls' in choice.message and choice.message.tool_calls:
                    current_tool_calls = choice.message.tool_calls

                    # For each current tool call, accumulate its arguments
                    for current_call in current_tool_calls:
                        if isinstance(current_call, dict) and 'index' in current_call:
                            idx = current_call['index']

                            # Find existing accumulated call with same index
                            existing_call = None
                            for acc_call in accumulated_data[choice_idx]['tool_calls']:
                                if (isinstance(acc_call, dict) and
                                        acc_call.get('index') == idx):
                                    existing_call = acc_call
                                    break

                            if existing_call:
                                # Accumulate function fields from current call
                                if ('function' in current_call and
                                        current_call['function']):
                                    if 'function' not in existing_call:
                                        existing_call['function'] = {}

                                    # Accumulate function.name
                                    if 'name' in current_call['function']:
                                        if 'name' not in existing_call['function']:
                                            existing_call['function']['name'] = ''
                                        existing_call['function']['name'] += current_call['function']['name']

                                    # Accumulate function.arguments
                                    if 'arguments' in current_call['function']:
                                        if 'arguments' not in existing_call['function']:
                                            existing_call['function']['arguments'] = ''
                                        existing_call['function']['arguments'] += current_call['function']['arguments']

                                # Update other fields with latest values
                                existing_call.update({k: v for k, v in current_call.items()
                                                      if k != 'function' and v})
                                if 'function' in current_call and current_call['function']:
                                    existing_call['function'].update({k: v for k, v in current_call['function'].items()
                                                                      if k not in ['arguments', 'name'] and v})
                            else:
                                # Add new tool call
                                accumulated_data[choice_idx]['tool_calls'].append(dict(current_call))

                    # Update choice with accumulated tool_calls
                    choice.message.tool_calls = accumulated_data[choice_idx]['tool_calls']
                elif accumulated_data[choice_idx]['tool_calls']:
                    # If current response has no tool_calls but we have
                    # accumulated tool_calls, restore them
                    choice.message.tool_calls = accumulated_data[choice_idx]['tool_calls']

                # Restore role if we have it
                if accumulated_data[choice_idx]['role'] and (not hasattr(choice.message, 'role') or not choice.message.role):
                    choice.message.role = accumulated_data[choice_idx]['role']

            # Handle logprobs accumulation (only if logprobs exists)
            try:
                if ('logprobs' in choice and choice.logprobs and
                        isinstance(choice.logprobs, dict) and 'content' in choice.logprobs):
                    current_logprobs_content = choice.logprobs['content']
                    if current_logprobs_content and isinstance(current_logprobs_content, list):
                        # Initialize logprobs content if not exists
                        if 'logprobs' not in accumulated_data[choice_idx]:
                            accumulated_data[choice_idx]['logprobs'] = {'content': []}
                        elif 'content' not in accumulated_data[choice_idx]['logprobs']:
                            accumulated_data[choice_idx]['logprobs']['content'] = []

                        # Extend the accumulated logprobs content array
                        accumulated_data[choice_idx]['logprobs']['content'].extend(current_logprobs_content)
            except (KeyError, AttributeError, TypeError):
                # logprobs field might not exist or be in unexpected format, safely skip
                pass

            # Always set accumulated logprobs if we have any
            if (accumulated_data[choice_idx]['logprobs']['content'] and
                    hasattr(choice, 'logprobs') and choice.logprobs):
                choice.logprobs['content'] = accumulated_data[choice_idx][
                    'logprobs']['content']

            # Handle finish_reason for n > 1 case
            if (n > 1 and hasattr(choice, 'finish_reason') and
                    choice.finish_reason and
                    choice.finish_reason != 'null'):
                accumulated_data[choice_idx]['finish_reason'] = \
                    choice.finish_reason
                accumulated_data[choice_idx]['finished'] = True

        # Handle n > 1 case: different strategies for different finish_reason
        if n > 1:
            # Count finished choices
            finished_count = sum(1 for data in accumulated_data.values()
                                 if isinstance(data, dict) and
                                 data.get('finished', False))

            # Find all finished choices in current packet
            finished_choices_in_packet = []
            for choice in choices:
                if (hasattr(choice, 'finish_reason') and
                        choice.finish_reason and
                        choice.finish_reason != 'null'):
                    choice_idx = (choice.index if hasattr(choice, 'index') and
                                  'index' in choice else 0)
                    finish_reason = choice.finish_reason
                    finished_choices_in_packet.append(
                        (choice_idx, finish_reason, choice))

            # No finish_reason in current packet: return as is
            if not finished_choices_in_packet:
                return True

            # Get finish_reason type from first finished choice
            first_finish_reason = finished_choices_in_packet[0][1]

            # For stop: wait all choices, then merge into one result
            if first_finish_reason == 'stop':
                if finished_count < n:
                    # Hide finish_reason until all finished
                    for choice in choices:
                        if (hasattr(choice, 'finish_reason') and
                                choice.finish_reason and
                                choice.finish_reason != 'null'):
                            choice.finish_reason = 'null'
                else:
                    # All finished: merge all choices into one result
                    for data in accumulated_data.values():
                        if isinstance(data, dict) and 'all_choices_sent' in data:
                            data['all_choices_sent'] = True

                    # Return final result with all choices
                    all_choices = []
                    # Sort by choice_idx to ensure correct order
                    sorted_items = sorted(
                        [(idx, data) for idx, data in accumulated_data.items()
                         if isinstance(data, dict) and 'finished' in data],
                        key=lambda x: x[0]
                    )

                    for choice_idx, data in sorted_items:
                        # Create a new choice object
                        final_choice_dict = {
                            'index': choice_idx,
                            'finish_reason': data['finish_reason']
                        }

                        # Create message
                        message_dict = {
                            'role': data['role'] if data['role'] else 'assistant'
                        }
                        if data['content']:
                            message_dict['content'] = (
                                data['content'] if isinstance(data['content'], str)
                                else data['content'])
                        if data['reasoning_content']:
                            message_dict['reasoning_content'] = data['reasoning_content']
                        if data['tool_calls']:
                            message_dict['tool_calls'] = data['tool_calls']

                        final_choice_dict['message'] = message_dict

                        # Add logprobs if present
                        if data['logprobs']['content']:
                            final_choice_dict['logprobs'] = {
                                'content': data['logprobs']['content']
                            }

                        all_choices.append(final_choice_dict)

                    # Update output choices with all accumulated choices
                    parsed_response.output.choices = all_choices

                    # Aggregate usage from all choice indices
                    if 'usage_by_index' in accumulated_data and accumulated_data[
                            'usage_by_index']:
                        aggregated_usage = {}
                        usage_by_idx = accumulated_data['usage_by_index']

                        # Sum output_tokens and recalculate total_tokens
                        total_output_tokens = 0
                        input_tokens = None
                        prompt_tokens_details = None

                        for idx, usage in usage_by_idx.items():
                            if 'output_tokens' in usage:
                                total_output_tokens += usage['output_tokens']
                            # input_tokens should be the same for all indices
                            if input_tokens is None and 'input_tokens' in usage:
                                input_tokens = usage['input_tokens']
                            # Keep prompt_tokens_details from any index
                            # (should be same)
                            if (prompt_tokens_details is None and
                                    'prompt_tokens_details' in usage):
                                prompt_tokens_details = usage[
                                    'prompt_tokens_details']

                        # Build aggregated usage
                        if input_tokens is not None:
                            aggregated_usage['input_tokens'] = input_tokens
                        aggregated_usage['output_tokens'] = total_output_tokens
                        if input_tokens is not None:
                            aggregated_usage['total_tokens'] = (
                                input_tokens + total_output_tokens)
                        if prompt_tokens_details is not None:
                            aggregated_usage['prompt_tokens_details'] = (
                                prompt_tokens_details)

                        # Update response usage with aggregated values
                        parsed_response.usage = aggregated_usage
            else:
                # For non-stop (e.g., tool_calls): output each choice separately
                responses_to_yield = []

                for choice_idx, finish_reason, choice in finished_choices_in_packet:
                    current_data = accumulated_data.get(choice_idx)
                    if (current_data is None or
                            current_data.get('all_choices_sent', False)):
                        continue

                    current_data['all_choices_sent'] = True

                    # Create a new response for this choice
                    if responses_to_yield:
                        # Clone the response for additional choices
                        new_response = copy.deepcopy(parsed_response)
                    else:
                        # Use the original response for the first choice
                        new_response = parsed_response

                    # Deep copy choice to avoid modifying accumulated_data
                    choice_copy = copy.deepcopy(choice)

                    # Set only this choice in the response
                    new_response.output.choices = [choice_copy]

                    # Update usage with this choice's output tokens
                    if (new_response.usage and
                            'usage_by_index' in accumulated_data and
                            choice_idx in accumulated_data['usage_by_index']):
                        current_usage = accumulated_data['usage_by_index'][
                            choice_idx]
                        if 'output_tokens' in current_usage:
                            new_response.usage['output_tokens'] = (
                                current_usage['output_tokens'])
                            if 'input_tokens' in current_usage:
                                new_response.usage['total_tokens'] = (
                                    current_usage['input_tokens'] +
                                    current_usage['output_tokens'])

                    responses_to_yield.append(new_response)

                # Return list of responses if we have any
                if responses_to_yield:
                    return responses_to_yield
                else:
                    return False

    return True


def merge_multimodal_single_response(parsed_response, accumulated_data, n=1):
    """Merge a single response chunk with accumulated data.

    Args:
        parsed_response: The response chunk to merge
        accumulated_data: Dictionary storing accumulated data for each choice
        n: Number of expected choices (default 1)

    Returns:
        bool: True if this response should be yielded, False if filtered
    """
    # Check if all choices have been sent (for n > 1 case)
    if n > 1 and accumulated_data:
        all_sent = any(data.get('all_choices_sent', False)
                       for data in accumulated_data.values())
        if all_sent:
            return False

    # Track usage for each choice index when n > 1
    # Each streaming packet contains usage info for one specific choice
    if (n > 1 and parsed_response.usage and 
            parsed_response.output and parsed_response.output.choices and
            len(parsed_response.output.choices) > 0):
        if 'usage_by_index' not in accumulated_data:
            accumulated_data['usage_by_index'] = {}

        # Get the choice index from the first (and typically only) choice in this packet
        try:
            first_choice = parsed_response.output.choices[0]
            choice_idx = first_choice.index if hasattr(
                first_choice, 'index') and 'index' in first_choice else 0

            # Store only output_tokens for this choice index
            if 'output_tokens' in parsed_response.usage:
                accumulated_data['usage_by_index'][choice_idx] = dict(
                    parsed_response.usage)
        except (KeyError, AttributeError, IndexError):
            pass

    # Handle output.text accumulation when choices is null
    if (parsed_response.output and
            hasattr(parsed_response.output, 'text') and
            (not parsed_response.output.choices or parsed_response.output.choices is None)):
        choice_idx = 0
        if choice_idx not in accumulated_data:
            accumulated_data[choice_idx] = {
                'content': '',
                'reasoning_content': '',
                'tool_calls': [],
                'logprobs': {'content': []},
                'finished': False,
                'finish_reason': None,
                'all_choices_sent': False,
                'role': None
            }
        # Accumulate text if not empty
        if parsed_response.output.text:
            accumulated_data[choice_idx]['content'] += parsed_response.output.text
        # Always set accumulated content back to response
        parsed_response.output.text = accumulated_data[choice_idx]['content']
        return True

    # Process each choice in the choices array
    if parsed_response.output and parsed_response.output.choices:
        choices = parsed_response.output.choices

        # Filter out empty choices array
        if not choices:
            return False

        for choice_enum_idx, choice in enumerate(choices):
            # Use choice.index if available, otherwise use enumerate index
            try:
                choice_idx = choice.index if hasattr(choice, 'index') and 'index' in choice else choice_enum_idx
            except (KeyError, AttributeError):
                choice_idx = choice_enum_idx

            # Initialize accumulated data for this choice if not exists
            if choice_idx not in accumulated_data:
                accumulated_data[choice_idx] = {
                    'content': '',
                    'reasoning_content': '',
                    'tool_calls': [],
                    'logprobs': {'content': []},
                    'finished': False,
                    'finish_reason': None,
                    'all_choices_sent': False,
                    'role': None
                }

            # Handle message field - create if null
            if not choice.message:
                # Create message object with accumulated data
                choice.message = {
                    'role': accumulated_data[choice_idx]['role'] if accumulated_data[choice_idx]['role'] else 'assistant',
                    'content': accumulated_data[choice_idx]['content']
                }
                if accumulated_data[choice_idx]['reasoning_content']:
                    choice.message['reasoning_content'] = accumulated_data[choice_idx]['reasoning_content']
                if accumulated_data[choice_idx]['tool_calls']:
                    choice.message['tool_calls'] = accumulated_data[choice_idx]['tool_calls']
            else:
                # Save role if present
                if hasattr(choice.message, 'role') and choice.message.role:
                    accumulated_data[choice_idx]['role'] = choice.message.role

                # Handle content accumulation
                if 'content' in choice.message:
                    current_content = choice.message.content
                    # Check if content is multimodal format
                    if isinstance(current_content, list):
                        # Handle multimodal content (array format)
                        # Initialize accumulated content as array if not already
                        if not isinstance(accumulated_data[choice_idx]['content'], list):
                            accumulated_data[choice_idx]['content'] = []

                        # Only process if current_content is not empty
                        if current_content:
                            # Ensure accumulated content list has enough elements
                            while len(accumulated_data[choice_idx]['content']) < len(current_content):
                                accumulated_data[choice_idx]['content'].append({'text': ''})

                            # Merge each content element
                            for content_idx, content_item in enumerate(current_content):
                                if isinstance(content_item, dict) and 'text' in content_item:
                                    if content_item['text']:
                                        # Accumulate text content
                                        accumulated_data[choice_idx]['content'][content_idx]['text'] += content_item['text']

                        # Always set accumulated content back to response
                        choice.message.content = accumulated_data[choice_idx]['content']
                    elif current_content:
                        # Handle regular content (string format)
                        # Initialize accumulated content as string
                        if isinstance(accumulated_data[choice_idx]['content'], list):
                            accumulated_data[choice_idx]['content'] = ''
                        # Accumulate content if not empty
                        accumulated_data[choice_idx]['content'] += current_content
                        # Set accumulated content back to response
                        choice.message.content = accumulated_data[choice_idx]['content']
                    elif not current_content and accumulated_data[choice_idx]['content']:
                        # Current content is empty but we have accumulated content, restore it
                        choice.message.content = accumulated_data[choice_idx]['content']

                # Handle reasoning_content accumulation
                if 'reasoning_content' in choice.message:
                    current_reasoning_content = choice.message.reasoning_content
                    if current_reasoning_content:
                        accumulated_data[choice_idx]['reasoning_content'] += current_reasoning_content
                # Always set the accumulated reasoning_content back if we
                # have any, even if current response doesn't have it
                if accumulated_data[choice_idx]['reasoning_content']:
                    choice.message.reasoning_content = accumulated_data[choice_idx]['reasoning_content']

                # Handle tool_calls accumulation
                if 'tool_calls' in choice.message and choice.message.tool_calls:
                    current_tool_calls = choice.message.tool_calls

                    # For each current tool call, accumulate its arguments
                    for current_call in current_tool_calls:
                        if isinstance(current_call, dict) and 'index' in current_call:
                            idx = current_call['index']

                            # Find existing accumulated call with same index
                            existing_call = None
                            for acc_call in accumulated_data[choice_idx]['tool_calls']:
                                if (isinstance(acc_call, dict) and
                                        acc_call.get('index') == idx):
                                    existing_call = acc_call
                                    break

                            if existing_call:
                                # Accumulate function fields from current call
                                if ('function' in current_call and
                                        current_call['function']):
                                    if 'function' not in existing_call:
                                        existing_call['function'] = {}

                                    # Accumulate function.name
                                    if 'name' in current_call['function']:
                                        if 'name' not in existing_call['function']:
                                            existing_call['function']['name'] = ''
                                        existing_call['function']['name'] += current_call['function']['name']

                                    # Accumulate function.arguments
                                    if 'arguments' in current_call['function']:
                                        if 'arguments' not in existing_call['function']:
                                            existing_call['function']['arguments'] = ''
                                        existing_call['function']['arguments'] += current_call['function']['arguments']

                                # Update other fields with latest values
                                existing_call.update({k: v for k, v in current_call.items()
                                                      if k != 'function' and v})
                                if 'function' in current_call and current_call['function']:
                                    existing_call['function'].update({k: v for k, v in current_call['function'].items()
                                                                      if k not in ['arguments', 'name'] and v})
                            else:
                                # Add new tool call
                                accumulated_data[choice_idx]['tool_calls'].append(dict(current_call))

                    # Update choice with accumulated tool_calls
                    choice.message.tool_calls = accumulated_data[choice_idx]['tool_calls']
                elif accumulated_data[choice_idx]['tool_calls']:
                    # If current response has no tool_calls but we have accumulated tool_calls, restore them
                    choice.message.tool_calls = accumulated_data[choice_idx]['tool_calls']

                # Restore role if we have it
                if accumulated_data[choice_idx]['role'] and (not hasattr(choice.message, 'role') or not choice.message.role):
                    choice.message.role = accumulated_data[choice_idx]['role']

            # Handle logprobs accumulation (only if logprobs exists)
            try:
                if ('logprobs' in choice and choice.logprobs and
                        isinstance(choice.logprobs, dict) and 'content' in choice.logprobs):
                    current_logprobs_content = choice.logprobs['content']
                    if current_logprobs_content and isinstance(current_logprobs_content, list):
                        # Initialize logprobs content if not exists
                        if 'logprobs' not in accumulated_data[choice_idx]:
                            accumulated_data[choice_idx]['logprobs'] = {'content': []}
                        elif 'content' not in accumulated_data[choice_idx]['logprobs']:
                            accumulated_data[choice_idx]['logprobs']['content'] = []

                        # Extend the accumulated logprobs content array
                        accumulated_data[choice_idx]['logprobs']['content'].extend(current_logprobs_content)
            except (KeyError, AttributeError, TypeError):
                # logprobs field might not exist or be in unexpected format, safely skip
                pass

            # Always set accumulated logprobs if we have any
            if (accumulated_data[choice_idx]['logprobs']['content'] and
                    hasattr(choice, 'logprobs') and choice.logprobs):
                choice.logprobs['content'] = accumulated_data[choice_idx][
                    'logprobs']['content']

            # Handle finish_reason for n > 1 case
            if (n > 1 and hasattr(choice, 'finish_reason') and
                    choice.finish_reason and
                    choice.finish_reason != 'null'):
                accumulated_data[choice_idx]['finish_reason'] = \
                    choice.finish_reason
                accumulated_data[choice_idx]['finished'] = True

        # Handle n > 1 case: different strategies for different
        # finish_reason
        if n > 1:
            # Count finished choices
            finished_count = sum(1 for data in accumulated_data.values()
                                 if isinstance(data, dict) and
                                 data.get('finished', False))

            # Find all finished choices in current packet
            finished_choices_in_packet = []
            for choice in choices:
                if (hasattr(choice, 'finish_reason') and
                        choice.finish_reason and
                        choice.finish_reason != 'null'):
                    choice_idx = (choice.index if hasattr(choice, 'index') and
                                  'index' in choice else 0)
                    finish_reason = choice.finish_reason
                    finished_choices_in_packet.append(
                        (choice_idx, finish_reason, choice))

            # No finish_reason in current packet: return as is
            if not finished_choices_in_packet:
                return True

            # Get finish_reason type from first finished choice
            first_finish_reason = finished_choices_in_packet[0][1]

            # For stop: wait all choices, then merge into one result
            if first_finish_reason == 'stop':
                if finished_count < n:
                    # Hide finish_reason until all finished
                    for choice in choices:
                        if (hasattr(choice, 'finish_reason') and
                                choice.finish_reason and
                                choice.finish_reason != 'null'):
                            choice.finish_reason = 'null'
                else:
                    # All finished: merge all choices into one result
                    for data in accumulated_data.values():
                        if isinstance(data, dict) and 'all_choices_sent' in data:
                            data['all_choices_sent'] = True

                    # Return final result with all choices
                    all_choices = []
                    # Sort by choice_idx to ensure correct order
                    sorted_items = sorted(
                        [(idx, data) for idx, data in accumulated_data.items()
                         if isinstance(data, dict) and 'finished' in data],
                        key=lambda x: x[0]
                    )

                    for choice_idx, data in sorted_items:
                        # Create a new choice object
                        final_choice_dict = {
                            'index': choice_idx,
                            'finish_reason': data['finish_reason']
                        }

                        # Create message
                        message_dict = {
                            'role': data['role'] if data['role'] else 'assistant'
                        }
                        if data['content']:
                            message_dict['content'] = (
                                data['content'] if isinstance(data['content'],
                                                              str)
                                else data['content'])
                        if data['reasoning_content']:
                            message_dict['reasoning_content'] = (
                                data['reasoning_content'])
                        if data['tool_calls']:
                            message_dict['tool_calls'] = data['tool_calls']

                        final_choice_dict['message'] = message_dict

                        # Add logprobs if present
                        if data['logprobs']['content']:
                            final_choice_dict['logprobs'] = {
                                'content': data['logprobs']['content']
                            }

                        all_choices.append(final_choice_dict)

                    # Update output choices with all accumulated choices
                    parsed_response.output.choices = all_choices

                    # Aggregate usage from all choice indices
                    if 'usage_by_index' in accumulated_data and accumulated_data[
                            'usage_by_index']:
                        aggregated_usage = {}
                        usage_by_idx = accumulated_data['usage_by_index']

                        # Sum output_tokens and recalculate total_tokens
                        total_output_tokens = 0
                        input_tokens = None
                        prompt_tokens_details = None

                        for idx, usage in usage_by_idx.items():
                            if 'output_tokens' in usage:
                                total_output_tokens += usage['output_tokens']
                            # input_tokens should be the same for all indices
                            if input_tokens is None and 'input_tokens' in usage:
                                input_tokens = usage['input_tokens']
                            # Keep prompt_tokens_details from any index
                            # (should be same)
                            if (prompt_tokens_details is None and
                                    'prompt_tokens_details' in usage):
                                prompt_tokens_details = usage[
                                    'prompt_tokens_details']

                        # Build aggregated usage
                        if input_tokens is not None:
                            aggregated_usage['input_tokens'] = input_tokens
                        aggregated_usage['output_tokens'] = total_output_tokens
                        if input_tokens is not None:
                            aggregated_usage['total_tokens'] = (
                                input_tokens + total_output_tokens)
                        if prompt_tokens_details is not None:
                            aggregated_usage['prompt_tokens_details'] = (
                                prompt_tokens_details)

                        # Update response usage with aggregated values
                        parsed_response.usage = aggregated_usage
            else:
                # For non-stop (e.g., tool_calls): output each choice
                # separately
                responses_to_yield = []

                for choice_idx, finish_reason, choice in finished_choices_in_packet:
                    current_data = accumulated_data.get(choice_idx)
                    if (current_data is None or
                            current_data.get('all_choices_sent', False)):
                        continue

                    current_data['all_choices_sent'] = True

                    # Create a new response for this choice
                    if responses_to_yield:
                        # Clone the response for additional choices
                        new_response = copy.deepcopy(parsed_response)
                    else:
                        # Use the original response for the first choice
                        new_response = parsed_response

                    # Deep copy choice to avoid modifying accumulated_data
                    choice_copy = copy.deepcopy(choice)

                    # Set only this choice in the response
                    new_response.output.choices = [choice_copy]

                    # Update usage with this choice's output tokens
                    if (new_response.usage and
                            'usage_by_index' in accumulated_data and
                            choice_idx in accumulated_data['usage_by_index']):
                        current_usage = accumulated_data['usage_by_index'][
                            choice_idx]
                        if 'output_tokens' in current_usage:
                            new_response.usage['output_tokens'] = (
                                current_usage['output_tokens'])
                            if 'input_tokens' in current_usage:
                                new_response.usage['total_tokens'] = (
                                    current_usage['input_tokens'] +
                                    current_usage['output_tokens'])

                    responses_to_yield.append(new_response)

                # Return list of responses if we have any
                if responses_to_yield:
                    return responses_to_yield
                else:
                    return False

    return True
